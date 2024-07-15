import jax
from jax import numpy as jnp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
# first define a system and its dynamics
# we will use an inverted pendulum again

#constants
g = 9.8 # gravity
m = l = 1 # mass and length of bob, pendulum
mu = 0.01 # friction coeff
dt = 0.01 # step size
nT = 1000 # time steps
maxit = 50 # allowed number of times to optimize
r = 1e-5 # input cost

# now we create a function that models the pendulum

# pendulum dynamics
def pendulum_dynamics(x, u):
    theta, theta_dot = x
    theta_ddot = (g / l * jnp.sin(theta) - mu / (m * l**2) * theta_dot + 1 / (m * l**2) * u)
    return jnp.array([theta_dot, theta_ddot])
pendulum_dynamics = jax.jit(pendulum_dynamics)

# RK4
def f(x, u, dt):
    k1 = pendulum_dynamics(x, u)
    k2 = pendulum_dynamics(x + 0.5 * k1 * dt, u)
    k3 = pendulum_dynamics(x + 0.5 * k2 * dt, u)
    k4 = pendulum_dynamics(x + k3 * dt, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
f = jax.jit(f)

# define variables
U = jnp.zeros((nT)) # input sequence (u_k for all k)
X = jnp.zeros((2,nT)) # state trajectory (x_k for all k)
target = jnp.array((jnp.pi,0)) # target state - straight up


#initial and boundary condition
x0 = jnp.array((0,1))
X = X.at[:,0].set(x0) # initial state

for _ in range(maxit):


    for i in range(0,nT-1):
        x = x.at[:,i+1].set(f(x[:,i], u[i],dt)) # simulate state





















# animation function
def animate(i):
    theta = x[0, i]  # angle in radians
    x_pos = l * jnp.sin(theta)
    y_pos = l * jnp.cos(theta)
    line.set_data([0, x_pos], [0, y_pos])
    return line,

# set up the figure, the axis, and the plot element
fig, ax = plt.subplots()
ax.set_xlim(-l - 0.1, l + 0.1)
ax.set_ylim(-l - 0.1, l + 0.1)
line, = ax.plot([], [], 'o-', lw=2)

# call the animator
ani = FuncAnimation(fig, animate, frames=nT, interval=20, blit=True)
print(x[0,:])
plt.show()