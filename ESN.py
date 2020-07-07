import numpy as np
import cupy as cp
import scipy as sp
import scipy.sparse
import os.path
import matplotlib.pyplot as plt

"""Build an Echo State Network"""
class ESN:
    # connection adjacency matrix W (nonlocal):
    W = np.zeros((1,1))
    # reservoir nodes' activations:
    x = np.zeros((1,1))
    # Each alpha acts as basically time between updates:
    alpha_matrix = np.zeros((1,1))
    # W_in has shape (N_x, N_u + 1)
    W_in = np.zeros((1,1))
    # W_fb has shape (N_x, N_u)
    W_fb = np.zeros((1,1))
    # W_out has shape (N_y, 1 + N_x + N_u)
    W_out = np.zeros((1,1))

    def __init__(self, N_x, N_u, N_y, sparsity,
                 x_initial, alpha_input, scaling_W,
                 scaling_W_in, scaling_W_fb):
        # num_rows and num_cols are scalars
        # input_W_in has shape (N_x, N_u + 1)
        # x_initial has shape (N_x, N_x)
        # alpha_input has shape (N_x, N_x)
        self.W = self.build_W(N_x, sparsity, scaling_W)#input_W
        self.W_in = self.build_W_in(N_x, N_u, scaling_W_in)#input_W_in
        self.W_fb = self.build_W_fb(N_x, N_u, scaling_W_fb)#input_W_fb
        self.x = x_initial
        self.alpha_matrix = alpha_input

    def build_W(self, N_x, sparsity, scaling_W):
        # N_x integer
        # sparsity between 0 and 1 inclusive
        # scaling_W >= 1
        if os.path.isfile('./W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity)+'.txt'):
            W = np.loadtxt('W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity)+'.txt')
        else:
            # Build sparse adjacency matrix W:
            W_unnormalized = sp.multiply(sp.random.choice((-1,1), size=(N_x,N_x)),
                                         sp.sparse.random(N_x,N_x, density = sparsity).todense())
            # Normalize by largest eigenvalue and additional scaling factor
            # to control decrease of spectral radius.
            spectral_radius = np.max(abs(np.linalg.eigvals(W_unnormalized)))
            print("SPECTRAL RADIUS IS IS "+str(spectral_radius))
            W = np.float32(np.multiply( scaling_W, np.divide(W_unnormalized,spectral_radius) ))
            np.savetxt('W_'+str(N_x)+'_'+str(N_x)+'_'+str(sparsity)+'.txt',W, fmt = '%.4f')
        return W

    def build_W_in(self, N_x, N_u, scaling_W_in):
        W_in = (1.0/scaling_W_in)*np.random.random((N_x, 1+N_u))
        return W_in

    def build_W_fb(self, N_x, N_u, scaling_W_fb):
        W_fb = (1.0/scaling_W_fb)*np.random.random((N_x, N_u))
        return W_fb

    def update_reservoir(self, u, x_nm1, y_n):
        # u is input at specific time
        #   u has shape (N_u + 1)
        # print("Shape 1: " +str(np.shape(np.matmul(self.W,x_nm1)))) #(30, 400000)
        # print("Shape 2: " +str(np.shape( self.W_in)))
        # print("Shape 3: " +str(np.shape( np.hstack((u,np.array([1]))))))
        # print(np.matmul(self.W,x_nm1)
        #                     + np.matmul(self.W_in, np.hstack((u,np.array([1])))))
        # See page 16 of Lukosevicius PracticalESN for feedback info.

        x_n_tilde = np.tanh(np.matmul(self.W,x_nm1)
                            + np.matmul(self.W_in, np.hstack((np.array([1]),u)))
                            + 0*np.matmul(self.W_fb, y_n))
        x_n = np.multiply((1-self.alpha_matrix), x_nm1) \
              + np.multiply(self.alpha_matrix, x_n_tilde)
        self.x = x_n

    def output_Y(self, u, x_nm1):
        one_by_one = np.array([1])
        # concatinated_matrix has dimension (231,)
        concatinated_matrix = np.hstack((np.hstack((one_by_one, u)),
                                        x_nm1))
        # print(np.shape(self.W_out))
        # print(np.shape(x_nm1))
        # print(np.shape(concatinated_matrix))
        # print("Shape of output")
        # print(np.shape(np.matmul(self.W_out, concatinated_matrix)))
        return np.matmul(self.W_out, concatinated_matrix)

    def calculate_W_out(self, Y_target, x, N_x, beta, train_start_timestep, num_timesteps_train):
        # see Lukosevicius Practical ESN eqtn 11
        # Using the usual linear regression
        # print(np.shape(np.array([np.real(np.matmul(X.transpose(),
        #                                np.linalg.inv(np.outer(X,X))))])))
        N_u = np.shape(Y_target)[0]
        # print(str(np.shape(np.ones((1, num_timesteps)))) + "," + str(np.shape(Y_target)) + "," + str(np.shape(x.transpose())))
        X = np.vstack((np.ones((1,num_timesteps_train-train_start_timestep)), Y_target, x.transpose()))
        # print("Shape of X is:"+str(np.shape(X)))
        # print("Shape of x is:"+str(np.shape(x)))
        # print("Shape of Y_target_windowed is:"+str(np.shape(Y_target)))
        # print("Shape of second term is:"+str(np.shape(np.matmul(X.transpose(),
        #                                np.linalg.inv(np.matmul(X,X.transpose()) + beta*np.identity(1+N_x+N_u))))))
        # Ridge Regression
        W_out = np.matmul(np.array(Y_target), np.matmul(X.transpose(),
                                                        np.linalg.inv(np.matmul(X,X.transpose()) + beta*np.identity(1+N_x+N_u))))
        # plt.figure()
        # plt.plot(X)
        # plt.show()
        #np.matmul(np.array(Y_target), np.linalg.pinv(X))
        # print("Shape of W_out is now:"+str(np.shape(W_out)))
        # print("W_out: "+str(W_out))
        self.W_out = W_out

