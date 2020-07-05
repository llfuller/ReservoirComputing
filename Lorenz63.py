import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

# Code directly from Wikipedia page on the "Lorenz System" with minor modifications

def run_L63(t_final, dt):
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def f(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, t_final, dt)

    states = odeint(f, state0, t) #shape is (time points=400,000, spatial dims = 3)

    np.savetxt('L63_States.txt',states)

def plot_L63():
    states = np.loadtxt('L63_States.txt')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.draw()
    plt.show()

def setup_ESN_params_L63():
    N_u = 3
    N_y = 3
    return [N_u, N_y]