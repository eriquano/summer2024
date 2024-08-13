import jax
import jax.numpy as jnp
from jax import jit

import configOC 
# start by defining a function to propagate vehicle dynamics (longitudinal)
# vehicle model and cost function taken from https://arxiv.org/pdf/1712.03719

# need vehicle constants. here i will use a tesla model S
m = 2089 # kg
rho = 1.225 # kg/m^3
A_f = 2.34 # m^2
g = 9.81 # m/s^2
c_r = 0.01 
c_d = 0.23
eta = 0.9
r_W = 0.35 # m
Torque_max = 1000 # N*m
vmax = 69.44 # m/s
accmax = 8.65 # m/s^2
accmin = -3 # m/s^2
Paux = 1500 # Watts, assumed to be constant
Trep = 0.1 # s, how often we replan
Thor = 10 # s, time horizon
Shor = 100 # m, longitude horizon
delta_v = 1 # m/s
delta_s_grid = 10 # m, grid discretization
delta_t_grid = 1 # s, grid discretization
delta_t_exp = 3 * delta_t_grid # s, expansion limit 
delta_s_exp = 3 * delta_s_grid # m, expansion limit
T_LC = 4 # s, lane change time
N_l = 3 # number of lanes

v_opt = (Paux/(rho*c_d*A_f))**(1/3) # optimal cruising velocity (17)

def road_angle(s): # sinusoidal road 
    alpha = 0.1 * jnp.sin(s)
    return alpha

def torque(v): # motor torque
    return Torque_max * (1 - v / vmax)

def angvel(v): # angular velocity of motor
    return v / r_W

def elevation(s):
    elev = 0.1 * -jnp.cos(s)
    return elev

def longitudinal_dynamics(x): # continuous longitudinal dynamics, from https://arxiv.org/pdf/1712.03719
    s, v = x    
    alpha = road_angle(s)
    T_m = torque(v)
    w_m = angvel(v)

    Fr = 0.5 * rho * c_d * A_f * v**2 + c_r * m * g * jnp.cos(alpha) + m * g * jnp.sin(alpha) # resistance force
    k = (2 * r_W * jnp.pi * w_m) / v
    Fm = (k * T_m * eta ** (jnp.sign(T_m))) / v # motor force

    v_dot = (Fm - Fr) / m
    return jnp.array([v, v_dot])

def rk4(f,x,dt): # numerical integration with RK4
    k1 = dt * f(x)
    k2 = dt * f(x + 0.5 * k1)
    k3 = dt * f(x + 0.5 * k2)
    k4 = dt * f(x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6

# now we should define cost function
# calculates energy used for some step in space
# from https://arxiv.org/pdf/1712.03719

# def energyCost(): 
#     s,v = x
#     T_m = torque(v)
#     w_m = angvel(v)
#     k = (2 * r_W * jnp.pi * w_m) / v

#     energy = ((k * T_m) / (2 * r_W * jnp.pi) + Paux/v) * ds
#     return energy 

def costest(s1, v1, s2, v2):
    #kinetic and potential energy diff
    E_k = m * (v2**2 - v1**2)/2
    E_p = m * g * (elevation(s2) - elevation(s1))
    est = E_k + E_p

    #rolling energy
    ds = 0.01
    s_vec = jnp.arange(s1,s2,ds)
    est += c_r * m * g * jnp.sum(jnp.cos(road_angle(s_vec)) * ds)


    est += (s2-s1) * ((.5 * rho * c_d * A_f) * v_opt**2 + Paux/v_opt)
    return est

# function to calculate heuristic from https://arxiv.org/pdf/1712.03719

def heuristic(s_i,s_f,v_i,v_f,ds):
    E_k = m * (v_f**2 - v_i**2)/2
    E_p = m * g * (elevation(s_f) - elevation(s_i))
    s_vec = jnp.arange(s_i,s_f,ds)
    W_roll = c_r * m * g * jnp.sum(jnp.cos(road_angle(s_vec)) * ds)

    W_tot = E_k + E_p + W_roll
    h_soa = W_tot * eta ** (-jnp.sign(W_tot))
    return h_soa # + W_AI                                     
 # ignoring W_AI for now because its confusing and h_soa works

# define object Node() that keeps track of state values at nodes in graph

# function to determine obstacle
def collision_check(nprime, obstacles):
    n_collision = jnp.copy(nprime)
    for n in nprime:
        for obs in obstacles:
            if n.s == obs.s and n.l == obs.l:
                n_collision.remove(n)
    return n_collision

class Node():
    def __init__(self):
        # node values ROUNDED TO GRID
        self.v = 0 # velocity
        self.t = 0 # time
        self.s = 0 # longitudinal dist
        self.l = 0 # lateral dist
                   
        # parent values
        self.vp = 0 
        self.tp = 0
        self.sp = 0
        self.lp = 0

        # remainder values (offset from grid)
        self.tr = 0
        self.sr = 0
        self.lr = 0

        # lane change direction
        self.l_dir = 0

        # costs
        self.g = float('inf') # cost-to-come
        self.f = float('inf') # n.g + heuristic(n)

        #temporary traveling value
        self.t_interval = 0
    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.v == other.v and self.t == other.t and
                    self.s == other.s and self.l == other.l and
                    self.vp == other.vp and self.tp == other.tp and
                    self.sp == other.sp and self.lp == other.lp and
                    self.tr == other.tr and self.sr == other.sr and
                    self.lr == other.lr and self.l_dir == other.l_dir and
                    self.g == other.g and self.f == other.f and
                    self.t_interval == other.t_interval)
        return False

    def __hash__(self):
        return hash((self.v, self.t, self.s, self.l, self.vp, self.tp,
                     self.sp, self.lp, self.tr, self.sr, self.lr, self.l_dir,
                     self.g, self.f, self.t_interval))

    def __repr__(self):
        return (f"Node(v={self.v}, t={self.t}, s={self.s}, l={self.l}, "
                f"vp={self.vp}, tp={self.tp}, sp={self.sp}, lp={self.lp}, "
                f"tr={self.tr}, sr={self.sr}, lr={self.lr}, l_dir={self.l_dir}, "
                f"g={self.g}, f={self.f}, t_interval={self.t_interval})")



# expansion function, algorithm 2
def expand(n,obstacles):
    v_i = n.v * delta_v
    v_f = jnp.arange(0, vmax, delta_v)
    n_lon = [Node() for _ in range(len(v_f))] # generate blank child nodes for each v_f
   
    # calculate information for each child (s,v,t,remainders)
    for i in range(len(n_lon)):
        n_temp = n_lon[i]
        n_temp.v = v_f[i]
        n_temp.vp = v_i
        v_avg = (n_temp.vp + n_temp.v) / 2
        if v_avg < delta_s_exp / delta_t_exp: # node is at time expansion limit
            t_interval = delta_t_exp 
            acc = (n_temp.vp - n_temp.v) / t_interval
            s_interval = (n_temp.vp**2 - n_temp.v**2) / (2 * acc)
        else: # node is at longitudinal distance expansion limit
            s_interval = delta_s_exp
            acc = (n_temp.vp**2 - n_temp.v**2) / (2 * s_interval)
            t_interval = (n_temp.vp - n_temp.v) / acc

        n_temp.t_interval = t_interval

        # acceleration checking
        if (acc > accmax) or (acc < accmin):
            n_lon.pop(i)
            continue

        # compute child s, sr values
        child_s = s_interval + n.sr
        child_s = child_s/delta_s_grid
        floor_child_s = jnp.floor(child_s)
        n_temp.s = floor_child_s + n.s
        n_temp.sr = child_s - floor_child_s
        # compute child t, tr values
        child_t = t_interval + n.tr
        child_t = child_t/delta_t_grid
        floor_child_t = jnp.floor(child_t)
        n_temp.t = floor_child_t + n.t
        n_temp.tr = child_t - floor_child_t       

        # TODO: figure out what the "final state" is for heuristi function, lateral motion
    nprime = jnp.copy(n_lon)
    for ns in n_lon:
        ns.g = n.g + costest(n.s+n.sr, v_i, ns.s+ns.r, ns.v+ns.vr)
        ns.f = ns.g #fix
    # lateral motion variants
    if jnp.mod(n.l,1) != 1: # lane change is happening
        for ns in nprime:
            LC_modifier = ns.t_interval/T_LC
            # progress current lane changing maneuver
            if ns.l_dir == "L":
                temp = ns.l + ns.lr
                temp = temp - LC_modifier
                ns.l = jnp.floor(temp)
                ns.lr = temp - ns.l
            elif ns.l_dir == "R":
                temp = ns.l + ns.lr
                temp = temp + LC_modifier
                ns.l = jnp.floor(temp)
                ns.lr = temp - ns.l
            ns.g = ns.g + 1 # cost of lane change (fix)
            ns.f = ns.f # + h(s,v) fix
    else: # create lane change variants
        if n.l > 1:
            nr = jnp.copy(n_lon)
            for ns in nr:
                ns.l_dir = "R"
                temp = ns.l + ns.lr
                temp = temp + LC_modifier
                ns.l = jnp.floor(temp)
                ns.lr = temp - ns.l
                ns.g = ns.g + 1 # cost of lane change (fix)
                ns.f = ns.f # + h(s,v) fix              
            nprime_set = set(nprime)
            nr_set = set(nr)
            nprime = list(nprime_set.union(nr_set))
        if n.l < N_l:
            nl = jnp.copy(n_lon)
            for ns in nl:
                ns.l_dir = "L"
                temp = ns.l + ns.lr
                temp = temp - LC_modifier
                ns.l = jnp.floor(temp)
                ns.lr = temp - ns.l
                ns.g = ns.g + 1 # cost of lane change (fix)
                ns.f = ns.f # + h(s,v) fix
            nprime_set = set(nprime)
            nl_set = set(nl)
            nprime = list(nprime_set.union(nl_set))
    n_collision = collision_check(nprime, obstacles)

# hybrid A* search, algorithm 1
grid = []
def A_star(start, end, obstacles): 
    '''initialize starting node with g=h=f=0 somewhere'''

    rowsTotal = len(grid[0])
    colsTotal = len(grid[0][0])
    timeStep = len(grid)

    openList = []
    closedList = [[[False for _ in range(colsTotal)] for _ in range(rowsTotal)] for _ in range(timeStep)]
    nodeInfo = [[[Node() for _ in range(colsTotal)] for _ in range(rowsTotal)] for _ in range(timeStep)]

