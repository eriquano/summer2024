import jax
import jax.numpy as jnp
from jax import grad


#-----------------------------------------------------------#

'''Nocetal Excercise 3.1
   Rosenbrock Function'''
import jax
import jax.numpy as jnp
from jax import grad

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# solve for p based on chosen method
def direction(direc,x):
    ret = 0
    if direc == 'Steep':
        return -grad(f)(x)
    elif direc == 'Newton':
        hess = jax.hessian(f)(x)
        return -jnp.linalg.solve(hess, grad(f)(x))
    

# define variables
direc = 'Steep' # chosen direction
iterations = 100 # iterations of descent
alpha = jnp.ones([iterations]) # initial step size guess
x = jnp.array([1.2, 1.2]) # initial point
rho = 0.5 # contraction factor 
c = 1e-4 # constant depending on algorithm

# iterate over points in descent
for i in range(iterations): # compute alpha, p for each point along descent
    p = direction(direc, x) # compute p based on gradient at x
    while f(x + alpha[i] * p) > f(x) + c * alpha[i] * jnp.dot(grad(f)(x),p): #backtracking line search
        alpha = alpha.at[i].set(alpha[i]*rho)
        print(f'intermediate alpha value: {alpha[i]}')
        print(i)
    x += alpha[i] * p # descend
    alpha = alpha.at[i+1].set(alpha[i]) # update alpha 
    print(f'Iteration {i} step size: {alpha[i]}')

