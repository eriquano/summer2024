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
# now we create a function that models the pendulum

# Pendulum dynamics function
def pendulum_dynamics(x, u):
    theta, theta_dot = x
    theta_ddot = (g / l * jnp.sin(theta) - mu / (m * l**2) * theta_dot + 1 / (m * l**2) * u)
    return jnp.array([theta_dot, theta_ddot])
pendulum_dynamics = jax.jit(pendulum_dynamics)

# Runge-Kutta 4th order method
def f(x, u, dt):
    k1 = pendulum_dynamics(x, u)
    k2 = pendulum_dynamics(x + 0.5 * k1 * dt, u)
    k3 = pendulum_dynamics(x + 0.5 * k2 * dt, u)
    k4 = pendulum_dynamics(x + k3 * dt, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


f = jax.jit(f)

# define variables
u = jnp.zeros((nT))
x = jnp.zeros((2,nT))
x = x.at[:,0].set(jnp.array((0, 1))) # initial state

for i in range(0,nT-1):
    x = x.at[:,i+1].set(f(x[:,i], u[i],dt)) # simulate state





















# Animation function
def animate(i):
    theta = x[0, i]  # angle in radians
    x_pos = l * jnp.sin(theta)
    y_pos = l * jnp.cos(theta)
    line.set_data([0, x_pos], [0, y_pos])
    return line,

# Set up the figure, the axis, and the plot element
fig, ax = plt.subplots()
ax.set_xlim(-l - 0.1, l + 0.1)
ax.set_ylim(-l - 0.1, l + 0.1)
line, = ax.plot([], [], 'o-', lw=2)

# Call the animator
ani = FuncAnimation(fig, animate, frames=nT, interval=20, blit=True)
print(x[0,:])
plt.show()