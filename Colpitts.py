import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

# Code directly from Wikipedia page on the "Lorenz System" with minor modifications

def run_Colpitts(t_final, dt):
    alpha = 5.0
    gamma = 0.08
    q = 0.7
    eta = 6.3

    def f(state, t):
        x1, x2, x3 = state  # Unpack the state vector
        return alpha*x2, -gamma*(x1+x3)-q*x2, eta*(x2+1-np.exp(-x1))  # Derivatives

    state0 = [0.1, 0.1, 0.1]
    t = np.arange(0.0, t_final, dt)

    states = odeint(f, state0, t) #shape is (time points=400,000, spatial dims = 3)

    np.savetxt('Colpitts_States.txt',states)

def plot_Colpitts():
    states = np.loadtxt('Colpitts_States.txt')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.draw()
    plt.show()

def setup_ESN_params_Colpitts():
    N_u = 3
    N_y = 3
    return [N_u, N_y]