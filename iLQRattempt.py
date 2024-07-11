import jax
from jax import numpy as jnp
import scipy as sp


# begin by defining the nonlinear dynamics of an inverted pendulum (example, can be anything)
# goal: make pendulum swing up

# constants
g = 9.8 # gravity
m = l = 1 # mass and length of bob, pendulum
mu = 0.01 # friction coeff
maxit = 100 # max iterations
nT = 200 # number of time steps 

@jax.jit
def f(x,u): # x[k+1] = f(x[k], u[k])

    newstate = jnp.array([[x[1]],[g / l * jnp.sin(x[0]) - mu / (m * l**2) * x[1] + 1 / (m * l**2) * u]])

    return jnp.squeeze(newstate)

# initialize a control sequence, state
x_star = jnp.array((jnp.pi,0)) # desired state
u = jnp.zeros((nT)) 
x = jnp.zeros((2,nT))
v = x

numcontrols = 1 # number of inputs
stsz = x.shape[0] # number of state variables                           
A = jnp.zeros((nT,stsz,stsz))
B = jnp.zeros((nT,stsz,numcontrols))

Q_f = Q = jnp.eye((2)) #scenario dependent
S = A
R = 1e-5 # scenario dependent

# boundary condition
S = S.at[-1,:,:].set(Q_f)

# begin iterating 
for i in range(maxit):
    delta_x = x_old = x

    # generate nominal trajectory (forward pass)
    for k in range(0, nT-1):
        x = x.at[:,k+1].set(f(x[:,k],u[k])) # this is x_k
        delta_x = delta_x.at[:,k+1].set(x[:,k+1] - x_old[:,k+1])
        A = A.at[k,:,:].set(jax.jacfwd(f,0)(x[:,k],u[k])) # D_x(f)
        B = B.at[k,:,:].set(jnp.expand_dims(jax.jacfwd(f,1)(x[:,k],u[k]),axis=-1)) # D_u(f)
    
    v = v.at[:,-1].set(jnp.dot(Q_f,x[:,-1]-x_star)) # boundary condition

    # solve for optimal controller variables (backward pass)
    K = jnp.zeros((nT,numcontrols,stsz))
    Kv = K
    Ku = jnp.expand_dims(u,axis=0)
    delta_u = jnp.expand_dims(u,axis=0)
    

    for j in range(nT-2,-1,-1):

        temp = (jnp.linalg.inv(B[j,:,:].T @ S[j+1,:,:] @ B[j,:,:] + R))
        K = K.at[j,:,:].set(temp * B[j,:,:].T @ S[j+1,:,:] @ A[j,:,:])
        Kv = Kv.at[j,:,:].set(temp * B[j,:,:].T)
        Ku = Ku.at[:,j].set(jnp.squeeze(temp * R))

        temp2 = A[j,:,:] - B[j,:,:] @ K[j,:,:]
        S = S.at[j,:,:].set(A[j,:,:] @ S[j+1,:,:] @ temp2 + Q)
        
        v = v.at[:,j].set(temp2.T @ v[:,j+1] - (R * K[j,:,:].T * u[j]).reshape((2,)) + Q @ x[:,j])
        
        delta_u = delta_u.at[:,j].set(-K[j,:,:] @ delta_x[:,j] - Kv[j,:,:] @ v[:,j+1] - Ku[:,j] * u[j])
        
    # create improved nominal control sequence
    
    u = u.at[:].add(jnp.squeeze(delta_u))
    
    # OK NOW CHECK COST

print(u)

#print(x)

x = x.at[:,0].set(jnp.array((jnp.pi/2, 0)))



