import jax
import jax.numpy as jnp
from jax import jit

# start by defining a function to propagate vehicle dynamics (longitudinal)

# need vehicle constants. here i will use a tesla model S
m = 2089 # kg
rho = 1.225 # kg/m^3
A_f = 2.34 # m^2
g = 9.81 # m/s^2
c_r = 0.01 
c_d = 0.23
eta = 0.9
r_W = 0.35 # m
Tmax = 1000 # N*m
vmax = 69.44 # m/s


def road_angle(x): # sinusoidal road 
    s = x[0]
    alpha = 0.1 * jnp.sin(s)
    return alpha

def torque(x): # motor torque
    v = x[1]
    return Tmax * (1 - v / vmax)

def angvel(x): # angular velocity of motor
    v = x[1]
    return v / r_W

def longitudinal_dynamics(x): # continuous longitudinal dynamics
    s, v = x    
    alpha = road_angle(x)
    T_m = torque(x)
    w_m = angvel(x)

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
