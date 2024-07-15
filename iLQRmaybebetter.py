import jax
from jax import numpy as jnp
from jax import jit,jacfwd
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
# first define a system and its dynamics
# we will use an inverted pendulum again

#constants
g = 9.8 # gravity
m = length = 1 # mass and length of bob, pendulum
mu = 0.01 # friction coeff
dt = 0.01 # step size
nT = 1000 # time steps
maxit = 100 # allowed number of times to optimize
lambda_factor = 10
lamb = 1. # for LM heuristic
lamb_max = 1000
r = 1e-5 # input cost
convergence_num = 1e-5 # threshold for convergence

# now we create a function that models the pendulum

# pendulum dynamics
def pendulum_dynamics(x, u):
    theta, theta_dot = x
    theta_ddot = (g / length * jnp.sin(theta) - mu / (m * length**2) * theta_dot + 1 / (m * length**2) * u)
    return jnp.array([theta_dot, theta_ddot])
pendulum_dynamics = jit(pendulum_dynamics)

# RK4
def f(x, u, dt):
    k1 = pendulum_dynamics(x, u)
    k2 = pendulum_dynamics(x + 0.5 * k1 * dt, u)
    k3 = pendulum_dynamics(x + 0.5 * k2 * dt, u)
    k4 = pendulum_dynamics(x + k3 * dt, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
f = jit(f)

# function that calculates immediate cost
def run_cost(x,u):
    l = jnp.sum(u**2) # example cost func - control is punished

    return l
run_cost = jit(run_cost)

# function that calculates final cost
def final_cost(x):
    wp = 1e4 # final position cost weight
    wv = 1e4 # final velocity cost weight
    theta = x[0]  # angle in radians
    x_pos = length * jnp.sin(theta)
    y_pos = length * jnp.cos(theta)

    target_theta = target[0]  # angle in radians
    target_x_pos = length * jnp.sin(target_theta)
    target_y_pos = length * jnp.cos(target_theta)
    pos_err = jnp.array([x_pos - target_x_pos,
                         y_pos - target_y_pos])
    
    l = (wp * jnp.sum(pos_err**2) + wv * (x[1]-target[1])**2) # final cost - state is punished
    return l
final_cost = jit(final_cost)

# simulation script
def simulate(x0,X,U_in):
    X = X.at[:,0].set(x0)
    cost = 0

    # simulate trajectory using current control sequence
    for i in range(nT-1):
        X = X.at[:,i+1].set(f(X[:,i], U_in[i],dt)) # simulate state
        l = run_cost(X[:,i], U_in[i])
        cost += dt * l
    # add final cost to running cost
    l_f = final_cost(X[:,-1])
    cost += l_f
    return X, cost
simulate = jit(simulate)

# define variables
U = jnp.zeros((nT)) # input sequence (u_k for all k)
X = jnp.zeros((2,nT)) # state trajectory (x_k for all k)
target = jnp.array((jnp.pi,0)) # target state - straight up

states = X.shape[0]
controls = 1

#initial and boundary condition
x0 = jnp.array((0,1))


#-------------ILQR BEGINS HERE-------------#
for _ in range(maxit):

    # forward pass using current control sequence
    X, cost = simulate(x0,X,U)
    old_cost = jnp.copy(cost)

    # define partial derivatives/jacobians
    f_x = jnp.zeros((nT,states,states))
    f_u = jnp.zeros((nT, states, controls))

    l = jnp.zeros((nT,1))
    l_x = jnp.zeros((nT, states))
    l_xx = jnp.zeros((nT, states, states))
    l_u = jnp.zeros((nT, controls))
    l_uu = jnp.zeros((nT,controls, controls))
    l_ux = jnp.zeros((nT, controls, states))

    # compute jacobians
    for i in range(nT-1):
        f_x = f_x.at[i].set(jit(jacfwd(f,0))(X[:,i], U[i], dt))
        f_u = f_u.at[i].set(jnp.expand_dims(jit(jacfwd(f,1))(X[:,i], U[i], dt), axis = -1))
        l = l.at[i].set(dt * run_cost(X[:,i],U[i]))
        l_x = l_x.at[i].set(dt * jit(jacfwd(run_cost,0))(X[:,i], U[i]))
        l_xx = l_xx.at[i].set(dt * jit(jacfwd(jacfwd(run_cost,0),0))(X[:,i], U[i]))
        l_u = l_u.at[i].set(dt * jit(jacfwd(run_cost,1))(X[:,i], U[i]))
        l_uu = l_uu.at[i].set(dt * jit(jacfwd(jacfwd(run_cost,1),1))(X[:,i], U[i]))
        l_ux = l_ux.at[i].set(dt * jit(jacfwd(jacfwd(run_cost,1),0))(X[:,i], U[i]))
    # make sure to include final values
    l = l.at[-1].set(final_cost(X[:,-1]))
    l_x = l_x.at[-1].set(jit(jacfwd(final_cost))(X[:,-1]))
    l_xx = l_xx.at[-1].set(jit(jacfwd(jacfwd(final_cost)))(X[:,-1]))

    # now we can optimize everything using these derivatives and performing a backward pass
    # starting at the final state

    # final values/optimal cost-to-go
    V = jnp.copy(l[-1])
    V_x = jnp.copy(l_x[-1])
    V_xx = jnp.copy(l_xx[-1])

    #initialize feed-forward and feedback gain
    k = jnp.zeros((nT,controls))
    K = jnp.zeros((nT, controls, states))

    # backward pass

    for i in range(nT-2,-1,-1):
        # pseudo hamiltonian expansion 
        Q_x = l_x[i] + jnp.dot(f_x[i].T,V_x)
        Q_u = l_u[i] + jnp.dot(f_u[i].T,V_x)
        Q_xx = l_xx[i] + jnp.dot(jnp.dot(f_x[i].T,V_xx), f_x[i]) # there is another term here, but since f_xx = f_xu = f_uu = 0, we omit
        Q_ux = l_ux[i] + jnp.dot(jnp.dot(f_u[i].T,V_xx), f_x[i])
        Q_uu = l_uu[i] + jnp.dot(jnp.dot(f_u[i].T,V_xx), f_u[i])

        # Levenberg-Marquardt heuristic calcs
        # necessary because in the next step we need inv(Q_uu), so we need to prevent danger
        Q_uu_CPU = jax.device_put(Q_uu, jax.devices('cpu')[0]) # JAX does not support nonsymmetric eigendecomp on GPU
        Q_uuvals, Q_uuvecs = jnp.linalg.eig(Q_uu_CPU)
        Q_uuvals = Q_uuvals.at[Q_uuvals < 0].set(0.0)
        Q_uuvals += lamb
        Q_uu_inv = jnp.dot(Q_uuvecs, jnp.dot(jnp.diag(1.0/Q_uuvals), Q_uuvecs.T))

        # feedback variables
        k = k.at[i].set(-jnp.dot(Q_uu_inv, Q_u))
        K = K.at[i].set(-jnp.dot(Q_uu_inv, Q_ux))

        # compute updated V derivatives for next iteration
        V_x = Q_x - jnp.dot(jnp.dot(K[i].T,Q_uu),k[i])
        V_xx = Q_xx - jnp.dot(jnp.dot(K[i].T,Q_uu),K[i])
        
    # initialize updated variables
    U_updated = jnp.zeros((nT)) # initialize new U
    x_new = jnp.copy(x0) # initialized new state

    # forward pass for new trajectory and control sequence
    for i in range(nT-1):
        U_updated = U_updated.at[i].set(jnp.squeeze(U[i] + k[i] + jnp.dot(K[i], x_new - X[:,i])))# calculate new control input
        x_new = f(x_new,U_updated[i],dt) # find next state in order to calc next u

    # calculate new trajectory X
    X_updated, cost_new = simulate(x0, X, U_updated)

    # LM heuristic
    if cost_new < cost:
        # good - we decrease lambda for a better inverse
        lamb /= lambda_factor

        # update
        X = jnp.copy(X_updated)
        U = jnp.copy(U_updated)
        old_cost = jnp.copy(cost)
        cost = jnp.copy(cost_new)

        if (abs(old_cost - cost) / cost) < convergence_num: # check for convergence
            break
    else:
        lamb *= lambda_factor
        if lamb > lamb_max:
            break




# animation function
def animate(i):
    theta = X[0, i]  # angle in radians
    x_pos = length* jnp.sin(theta)
    y_pos = length * jnp.cos(theta)
    line.set_data([0, x_pos], [0, y_pos])
    return line,
animate = jit(animate)

# set up the figure, the axis, and the plot element
fig, ax = plt.subplots()
ax.set_xlim(-length- 0.1, length + 0.1)
ax.set_ylim(-length - 0.1, length + 0.1)
line, = ax.plot([], [], 'o-', lw=2)

# call the animator
ani = FuncAnimation(fig, animate, frames=nT, interval=20, blit=True)
print(X[0,:])
plt.show()