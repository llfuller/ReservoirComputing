import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# original NaKL code used as basis for this script typed by Randall Clark
# No idea what the units are
#NaKL model
def run_NaKL(t_final, dt, time_sequence, I_ext):
    gNa = 120
    ENa = 50
    gK = 20
    EK = -77
    gL = 0.3
    EL = -54.4
    Vm1 = -40.0
    dVm = 15
    taum0 = .1
    taum1 = .4
    Vh0 = -60
    dVh = -15
    tauh0 = 1
    tauh1 = 7
    Vn1 = -55
    dVn = 30
    taun0 = 1
    taun1 = 5
    #I_ext = 0.0001*np.ones((float(t_final)/dt)) #numpy array of 0.0001 with dimension number of timesteps
    def f(state, t):
        v0,m00,h00,n00 = state[:4]
        I_ext_t = np.interp(t, time_sequence, I_ext)
        dvdt = gK*n00**4*(EK - v0) + gL*(EL - v0) + gNa*h00*m00**3*(ENa - v0) + I_ext_t
        dmdt = (-m00 + 0.5*np.tanh((-Vm1 + v0)/dVm) + 0.5)/(taum0 + taum1*(1 - np.tanh((-Vm1 + v0)/dVm)**2))
        dhdt = (-h00 + 0.5*np.tanh((-Vh0 + v0)/dVh) + 0.5)/(tauh0 + tauh1*(1 - np.tanh((-Vh0 + v0)/dVh)**2))
        dndt = (-n00 + 0.5*np.tanh((-Vn1 + v0)/dVn) + 0.5)/(taun0 + taun1*(1 - np.tanh((-Vn1 + v0)/dVn)**2))
        dXdt = [dvdt, dmdt, dhdt, dndt, 0] #I_ext is not meant to change, so I just put a zero in its place
        return dXdt

    state0 = [-50, 0.4, 0.4, 0.4, 0]
    t = np.arange(0.0, t_final, dt)
    states = odeint(f, state0, t) #shape is (time points=400,000, spatial dims = 3)
    print("State shape:"+str(np.shape(states)))
    print("times: "+str(t))
    states[:,4] = I_ext[:np.shape(states)[0]]
    np.savetxt('NaKL_States.txt',states, fmt = '%.4f')

def plot_NaKL(time_sequence, I_ext_array):
    states = np.loadtxt('NaKL_States.txt')
    plt.plot(time_sequence,states[:, 0])
    plt.plot(time_sequence, I_ext_array)
    plt.show()

def setup_ESN_params_NaKL():
    N_u = 5
    N_y = 5
    return [N_u, N_y]

# t_final = 10000
# dt = 0.01
# time_sequence = np.arange(0.0, t_final, dt)
# def I_ext(t):
#     return 10+100*0.25*np.sin(0.01*t)
# I_ext_array = np.array([I_ext(t) for t in time_sequence]) # I do this because real stimuli are discrete and I don't want to recalculate L63 from the beginning for every timestep
#
#
# run_NaKL(t_final, dt, time_sequence, I_ext_array)
# plot_NaKL(time_sequence, I_ext_array)
#
# t_final = 1000.0
# dt = 0.01
# num_timesteps_sim = int(round(float(t_final/dt)))
# time_sequence = np.arange(0.0, t_final, dt)
# I_L63 = np.loadtxt("L63_States_slowX10.txt")[:num_timesteps_sim,0]#[10+100*0.25*np.sin(0.01*t) for t in time_sequence] #
# run_NaKL(t_final, dt, time_sequence, I_L63)
# plot_NaKL(time_sequence, I_L63)