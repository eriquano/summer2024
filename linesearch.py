import jax
import jax.numpy as jnp
from jax import grad

# #1D function to minimize
# def f(x,alpha,p):
#     #p is the descent direction, alpha is the step length, x is an x value along descent


#     return 

# #univariate function 
# def phi(alpha):
#     return f(x,alpha,p)

# #sufficient decrease condition:
# '''phi(alpha_k) <= phi(0) +c*alpha_k*phi'(0)'''

# #if given guess alpha_0, if we have phi(alpha_0) <= phi(0) +c*alpha_k*phi'(0), condition is satisfied and we end search
# #if not, we know [0, alpha_0] contains acceptable step length


# #quadratic approx phi_q(alpha) of phi(alpha):
# #KNOWN: phi(0), phi'(0), phi(alpha_0)
# def phi_q(alpha):
#     diff = (phi(alpha_0) - phi(0) - alpha_0 * grad(phi)(0)) / alpha_0**2
#     return diff * alpha**2 + grad(phi)(0) * alpha + phi(0)

# #test new trial alpha: alpha_1, defined as minimizer of phi_q

# alpha_1 = -(grad(phi)(0) * alpha_0**2) / (2 * (phi(alpha_0) - phi(0) - grad(phi)(0) * alpha_0)) 

# #if this meets SDC, terminate search
# #otherwise create cubic function for interpolation again
# abdot = jnp.dot(jnp.array([[alpha_0**2, -alpha_1**2],[-alpha_0**3, alpha_1**3]]), jnp.array([[phi(alpha_1) - phi(0) - grad(phi)(0) * alpha_1], [phi(alpha_0) - phi(0) - grad(phi)(0) * alpha_0]]))
# [a, b] = (1 / (alpha_0**2 * alpha_1**2 * (alpha_1 - alpha_0))) * abdot

# def phi_c(alpha):
#     return a * alpha**3 + b * alpha**2 + alpha * grad(phi)(0) + phi(0) #might not be necessary to compute

# #differentiating phi_c reveals that alpha_2 minimizer of phi_c is between [0,alpha_1] and is given by:
# alpha_2 = (-b + jnp.sqrt(b**2 - 3 * a * grad(phi)(0))) / (3*a)

# #this process is repeated if needed using cubic interpolant of phi(0), phi'(0), and 2 most recent phi until satisfying alpha is located

# '''if any alpha_i is too close to the previous or too much smaller than previous, we reset alpha_i = alpha_i-1 / 2'''


# #OR CUBIC INTERPOLATION THE WHOLE TIME

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

