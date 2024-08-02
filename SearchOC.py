import jax
import jax.numpy as jnp
from jax import jit

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
Paux = 1500 # Watts, assumed to be constant
Trep = 0.1 # s
Thor = 10 # s
Shor = 100 # m

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

def longitudinal_dynamics(x): # continuous longitudinal dynamics
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

def energyCost(x,ds):
    s,v = x
    T_m = torque(v)
    w_m = angvel(v)
    k = (2 * r_W * jnp.pi * w_m) / v

    energy = ((k * T_m) / (2 * r_W * jnp.pi) + Paux/v) * ds
    return energy

# function to calculate heuristic

def h(s_i,s_f,v_i,v_f,ds):
    E_k = m * (v_f**2 - v_i**2)/2
    E_p = m * g * (elevation(s_f) - elevation(s_i))
    s_vec = jnp.arange(s_i,s_f,ds)
    W_roll = c_r * m * g * jnp.sum(jnp.cos(road_angle(s_vec)) * ds)

    W_tot = E_k + E_p + W_roll
    h_soa = W_tot * eta ** (-jnp.sign(W_tot))
    return h_soa # + W_AI                                     
 # ignoring W_AI for now because its confusing and h_soa works

# define object Node() that keeps track of state values at nodes in graph

class Node():
    def __init__(self):
        self.v = 0
        self.t = 0
        self.s = 0
        self.l = 0

        # parent values
        self.vp = 0
        self.tp = 0
        self.sp = 0
        self.lp = 0

        # remainder values
        self.tr = 0
        self.sr = 0
        self.lr = 0

        # lane change direction
        self.l_dir = 0

        # costs
        self.g = 0
        self.f = 0 # n.g + h(n)

