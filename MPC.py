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

# note: this code does not apply generally. small modifications need to be made to the weight/O,M matrices to handle MIMO

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
inputs = Bc.shape[1]
dt = 0.1 # length of time step
A = jnp.linalg.inv(jnp.eye(states) - dt * Ac)
B = dt * jnp.dot(A, Bc)

# now we have x[k] = Ax[k-1] + Bu[k-1]

# generate initial state and input
t = 300 # total time
x0 = jnp.zeros((states, 1))

# define desired trajectory. here we will use exponential
timeArray = jnp.linspace(0,t, int(t/dt)+1)
#trajectory = jnp.ones(int(t/dt)+1) - jnp.exp(-0.01*timeArray) 

trajectory = jnp.zeros((len(timeArray)))
trajectory = trajectory.at[0:1000].set(jnp.ones(1000)) #pulse trajectory option for fun
trajectory = trajectory.at[2001:].set(jnp.ones(1000))


# arrays to store states, inputs and outputs over at each step
X = jnp.zeros((states,len(timeArray)))
X = X.at[:,0].set(x0.flatten())
Y = jnp.zeros((C.shape[0], len(timeArray)))
Y= Y.at[:,0].set(jnp.dot(C,x0).flatten())
U = jnp.zeros((inputs,len(timeArray)))

# create weight matrices
f = 20 # prediction horizon
v = 20 # control horizon (where u changes)
C_rows = C.shape[0]

W1 = jnp.zeros((v * inputs, v * inputs))
W1 = W1.at[0,0].set(jnp.eye(inputs).item())
for i in range(1, v):
    W1 = W1.at[i, i].set(jnp.eye(inputs).item())
    W1 = W1.at[i, i-1].set(-jnp.eye(inputs).item())

Q0 = 0.0000000011
Q_else = 0.0001 #taken from the guide. these are user selected
W2 = jnp.zeros((v * inputs, v * inputs))
Q_diag = Q_else * jnp.ones(v * inputs)
Q_diag = Q_diag.at[0].set(Q0)
W2 = W2.at[jnp.diag_indices(v * inputs)].set(Q_diag)

W3 = jnp.dot(jnp.dot(W1.T,W2), W1)

W4 = jnp.zeros((f * C_rows,f * C_rows))
P = 10 # from guide, user selected
P_diag = P * jnp.ones(f * C_rows)
W4 = W4.at[jnp.diag_indices(f * C_rows)].set(P_diag)

# lifted matrix O
O=jnp.zeros(shape=(f*C_rows,states))

# create lifted matrices
O = jnp.zeros((C_rows * f, states))

for i in range(1, C_rows * f + 1):
    tempA = jnp.linalg.matrix_power(A,i)
    O = O.at[i-1,:].set(jnp.reshape(jnp.dot(C,tempA),(states,)))


M = jnp.zeros((C_rows * f, v * inputs))            
# start with elements until control horizon
for i in range(C_rows * f):
    if (i < v):
        for j in range(i+1):
            if j == 0:
                exp = jnp.eye(states)
            else:
                exp = jnp.dot(exp, A)
            M = M.at[i, i-j].set(jnp.dot(C, jnp.dot(exp,B)).item())
    else: # elements from control horizon to prediction horizon
        for j in range(v):
            if j==0: # last element
                A_bar = jnp.zeros((states, states))
                for k in range(i-v+2):
                    if k == 0:
                        exp = jnp.eye(states)
                    else:
                        exp = jnp.dot(exp,A)
                    A_bar += exp  # sum over elements in A_bar (A^f-v + A^f-v-1 and so on...)
                M = M.at[i, v-1].set(jnp.dot(C, jnp.dot(A_bar,B)).item())
            else:
                exp = jnp.dot(exp,A)
                M = M.at[i, v-j-1].set(jnp.dot(C, jnp.dot(exp,B)).item())


# calculate gain matrix that minimizes J
K = jnp.dot(jnp.dot(jnp.linalg.inv(jnp.dot(jnp.dot(M.T,W4),M) + W3), M.T), W4)

# function to propagate dynamics after every step

def dynamics(x,u):
    xnext = jnp.reshape(jnp.dot(A,x),shape=(states,inputs)) + jnp.dot(B,u)
    ynext = jnp.dot(C,x)
    return xnext, ynext

#----------MPC LOOP----------#

for i in range(len(timeArray)-f):
    # first establish the trajectory over the control horizon
    desiredTraj = trajectory[i:i+f]

    # compute s (diff between desired and actual trajectory)
    s = desiredTraj - jnp.dot(O,X[:,i])

    # compute control sequence
    sequence = jnp.dot(K, s)

    # apply first entry of sequence to state and propagate dynamics
    xnext, ynext = dynamics(X[:,i], sequence[0])

    #update next time step values
    X = X.at[:,i+1].set(xnext.flatten())
    Y = Y.at[:,i+1].set(ynext)
    U = U.at[:,i].set(sequence[0])

# plot
fig, ax = plt.subplots()
ax.set_xlim((0, t))
ax.set_ylim((0, jnp.max(Y[0,:])))
line1, = ax.plot([], [], 'g-', label='Desired Trajectory')
line2, = ax.plot([], [], 'r-', label='Controlled Trajectory')
ax.legend()

# split the frames into two phases
split_frame = len(timeArray) // 2

def update(frame):
    if frame < split_frame:
        line1.set_data(timeArray[:frame], trajectory[:frame])
        line2.set_data([], [])
    else:
        line1.set_data(timeArray, trajectory)
        line2.set_data(timeArray[:frame - split_frame], Y[0, :frame - split_frame])
    return line1, line2

# create animation
animation = ani.FuncAnimation(fig, update, frames=len(timeArray) + split_frame, blit=True, interval=1)
animation.save('trajectory_animation2.mp4', writer='ffmpeg', fps=60)  

plt.show()

# typo typoy