import jax
from jax import jacfwd, jit, vmap
from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# x[k+1] = Ax[k] + Bu[k] 
# z[k+1] = Cx[k+1]

# we want to determine control inputs u[k+i], i = 0,1,2,3, ..., v-1
# at t = k that will make the output z[k+1] follow 

# first define a system to test MPC. we will use mass-spring damper from aleksandar habar's example

# system parameters
m1 = 20
m2 = 20
k1 = 1000
k2 = 2000
d1 = 1
d2 = 5

Ac = jnp.array([[0, 1, 0, 0],[-(k1 + k2)/m1, -(d1 + d2)/m1, k2/m1, d2/m1], [0, 0, 0, 1], [k2/m2, d2/m2, -k2/m2, -d2/m2]])
Bc = jnp.array([[0, 0, 0, 1/m2]]).T
C = jnp.array([[1, 0, 0, 0]])

# discretize system (backwards euler)
states = Ac.shape[0]
dt = 0.01
A = jnp.linalg.inv(jnp.eye(states) - dt * Ac)
B = dt * jnp.dot(A, Bc)

# now we have x[k] = Ax[k-1] + Bu[k-1]

# generate initial state and input
t = 100
x0 = jnp.zeros((states, 1))
u = 10*jnp.ones((1,t))

# initialize matrix of states and matrix of outputs
X = jnp.zeros((states,t))
Y = jnp.zeros((C.shape[0], t))


# simulation function

def simulate(A, B, C, X, Y, x0, u):
    time = u.shape[1]
    X = X.at[:,0].set(x0.flatten())
    Y = Y.at[:,0].set(jnp.dot(C,x0).flatten())

    for i in range(1,time):
        X = X.at[:,i].set(jnp.dot(A,X[:,i-1]) + jnp.dot(B, u[:,i-1]).flatten())
        Y = Y.at[:,i].set(jnp.dot(C,X[:,i]).flatten())

    return X, Y
simulate = jit(simulate)

# now let's get to the real MPC stuff

f = 20 # prediction horizon
v = 20 # control horizon (where u changes)

# need to form lifted matrices O and M
C_rows = C.shape[0]
O = jnp.zeros((C_rows * f, states))
tempA = jnp.copy(A)

for i in range(C_rows * f):
    tempA = jnp.linalg.matrix_power(tempA,i)
    O = O.at[i,:].set(jnp.squeeze(jnp.dot(C,tempA)))
    


print(O)
M = jnp.zeros((C_rows*f))
























# plotting

fig, ax = plt.subplots()
ax.set_xlim(0, t)
ax.set_ylim(jnp.min(Y), jnp.max(Y))
line, = ax.plot([], [], lw=2)

def animate(i):
    x = jnp.arange(i)
    y = Y[0, :i]
    line.set_data(x, y)
    return line,

# call the animator
ani = ani.FuncAnimation(fig, animate,
                              frames=t, interval=dt*1000, blit=True)

# display the animation
plt.show()