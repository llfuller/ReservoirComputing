from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# Code directly from Wikipedia page on the "Lorenz 96 Model" with minor modifications


def run_L96(dims, t_final, dt):
    # These are our constants
    N = dims  # Number of variables, default 36
    F = 8  # Forcing

    def lorenz96(x, t):
        """Lorenz 96 model."""
        # Compute state derivatives
        d = np.zeros(N)
        # First the 3 edge cases: i=1,2,N
        d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
        d[1] = (x[2] - x[N-1]) * x[0] - x[1]
        d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
        # Then the general case
        for i in range(2, N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        # Add the forcing term
        d = d + F

        # Return the state derivatives
        return d

    x0 = F * np.ones(N)  # Initial state (equilibrium)
    x0[2] += 0.01  # Add small perturbation to 20th variable, default index 19
    t = np.arange(0.0, t_final, dt)

    x = odeint(lorenz96, x0, t)
    np.savetxt('L96_States.txt', x, fmt = '%.4f')

def plot_L96():
    x = np.loadtxt('L96_States.txt')
    # Plot the first three variables
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    plt.show()