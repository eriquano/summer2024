import jax
from jax import numpy as jnp
import scipy as sp
import matplotlib.pyplot as plt


# begin by defining the nonlinear dynamics of an inverted pendulum (example, can be anything)
# goal: make pendulum swing up

# constants
g = 9.8 # gravity
m = l = 1 # mass and length of bob, pendulum
mu = 0.01 # friction coeff
maxit = 100 # max iterations
nT = 200 # number of time steps 
threshold = 1e-5 # threshold cost difference

@jax.jit
def f(x,u): # x[k+1] = f(x[k], u[k])

    newstate = jnp.array([
        [x[1]],
        [g / l * jnp.sin(x[0]) - mu / (m * l**2) * x[1] + 1 / (m * l**2) * u]])

    return jnp.squeeze(newstate)

@jax.jit
def arraycalcs(A,B,S,R,Q,K,Kv,Ku,v,u,delta_u,delta_x,x,j):

    temp = (jnp.linalg.inv(B[j,:,:].T @ S[j+1,:,:] @ B[j,:,:] + R))
    K = K.at[j,:,:].set(temp * B[j,:,:].T @ S[j+1,:,:] @ A[j,:,:])
    Kv = Kv.at[j,:,:].set(temp * B[j,:,:].T)
    Ku = Ku.at[:,j].set(jnp.squeeze(temp * R))

    temp2 = A[j,:,:] - B[j,:,:] @ K[j,:,:]
    S = S.at[j,:,:].set(A[j,:,:].T @ S[j+1,:,:] @ temp2 + Q)
    
    v = v.at[:,j].set(temp2.T @ v[:,j+1] - (R * K[j,:,:].T * u[j]).reshape((2,)) + Q @ x[:,j])
    
    delta_u = delta_u.at[:,j].set(-K[j,:,:] @ delta_x[:,j] - Kv[j,:,:] @ v[:,j+1] - Ku[:,j] * u[j])
    return K, Kv, Ku, S, v, delta_u


# initialize a control sequence, state
x_star = jnp.array((jnp.pi,0)) # desired state
u = jnp.zeros((nT)) 
x = jnp.zeros((2,nT))
v = jnp.zeros_like(x)
x = x.at[:,0].set(jnp.array((jnp.pi/2, 0))) #optional starting point


numcontrols = 1 # number of inputs
stsz = x.shape[0] # number of state variables                           
A = jnp.zeros((nT,stsz,stsz))
B = jnp.zeros((nT,stsz,numcontrols))

Q_f = Q = jnp.eye((2)) #scenario dependent
S = jnp.zeros_like(A)
R = 1e-5 # scenario dependent

# boundary condition
S = S.at[-1,:,:].set(Q_f)

# initial cost
#cost = 0.5 * (x[:, -1] - x_star).T @ Q_f @ (x[:, -1] - x_star) + 0.5 * jnp.sum(jnp.einsum('ij,jk,ik->i', x.T, Q, x.T)) + 0.5 * jnp.sum(u**2 * R)
cost = 0.5 * (x[0,-1]**2 + x[1,-1**2]) + 0.5 * jnp.sum(R * u * u)
# begin iterating (something is wrong here)
for i in range(maxit):

    cost_old = cost
    delta_x = jnp.zeros_like(x)
    x_old = jnp.copy(x)

    # generate nominal trajectory (forward pass)
    for k in range(0, nT-1):
        x = x.at[:,k+1].set(f(x[:,k],u[k])) # this is x_k
        delta_x = delta_x.at[:,k+1].set(x[:,k+1] - x_old[:,k+1])
        A = A.at[k,:,:].set(jax.jacfwd(f,0)(x[:,k],u[k])) # D_x(f)
        B = B.at[k,:,:].set(jnp.expand_dims(jax.jacfwd(f,1)(x[:,k],u[k]),axis=-1)) # D_u(f)
    
    v = v.at[:,-1].set(jnp.dot(Q_f,x[:,-1]-x_star)) # boundary condition

    # solve for optimal controller variables (backward pass)
    K = jnp.zeros((nT,numcontrols,stsz))
    Kv = jnp.zeros_like(K)
    Ku = jnp.expand_dims(u,axis=0)
    delta_u = jnp.expand_dims(u,axis=0)
    
    for j in range(nT-2,-1,-1):
        K, Kv, Ku, S, v, delta_u = arraycalcs(A,B,S,R,Q,K,Kv,Ku,v,u,delta_u,delta_x,x,j)
        
    # create improved nominal control sequence and apply it
    u = u.at[:].add(jnp.squeeze(delta_u)) # improved control
    for k in range(0, nT-1):
        x = x.at[:,k+1].set(f(x[:,k],u[k])) # new x

    # check cost

    #cost = 0.5 * (x[:, -1] - x_star).T @ Q_f @ (x[:, -1] - x_star) + 0.5 * jnp.sum(jnp.einsum('ij,jk,ik->i', x.T, Q, x.T)) + 0.5 * jnp.sum(u**2 * R)
    cost = 0.5 * (x[0,-1]**2 + x[1,-1**2]) + 0.5 * jnp.sum(R * u * u)

    if jnp.abs((cost - cost_old) / cost_old) < threshold:
        break
    
# plot

fig, axs = plt.subplots(2)
fig.suptitle('angular position and angular velocity vs time')
axs[0].plot(jnp.arange(0,nT), x[0,:])
axs[1].plot(jnp.arange(0,nT), x[1,:])
plt.show()

# extra

#cost = 0.5 * (x[:,-1] - x_star).T @ Q_f @ (x[:,-1] - x_star) + 0.5 * jnp.sum(x.T @ Q @ x + u.T * R * u) old cost implementation
#x = x.at[:,0].set(jnp.array((jnp.pi/2, 0)))
# temp = (jnp.linalg.inv(B[j,:,:].T @ S[j+1,:,:] @ B[j,:,:] + R))
# K = K.at[j,:,:].set(temp * B[j,:,:].T @ S[j+1,:,:] @ A[j,:,:])
# Kv = Kv.at[j,:,:].set(temp * B[j,:,:].T)
# Ku = Ku.at[:,j].set(jnp.squeeze(temp * R))

# temp2 = A[j,:,:] - B[j,:,:] @ K[j,:,:]
# S = S.at[j,:,:].set(A[j,:,:] @ S[j+1,:,:] @ temp2 + Q)

# v = v.at[:,j].set(temp2.T @ v[:,j+1] - (R * K[j,:,:].T * u[j]).reshape((2,)) + Q @ x[:,j])

# delta_u = delta_u.at[:,j].set(-K[j,:,:] @ delta_x[:,j] - Kv[j,:,:] @ v[:,j+1] - Ku[:,j] * u[j])


#a change