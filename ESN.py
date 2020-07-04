import numpy as np
import scipy.sparse

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
    # W_out has shape (N_y, 1 + N_x + N_u)
    W_out = np.zeros((1,1))

    def __init__(self, N_x, N_u, N_y, sparsity,
                 x_initial, alpha_input, scaling_W,
                 scaling_W_in):
        # num_rows and num_cols are scalars
        # input_W_in has shape (N_x, N_u + 1)
        # x_initial has shape (N_x, N_x)
        # alpha_input has shape (N_x, N_x)
        self.W = self.build_W(N_x, sparsity, scaling_W)#input_W
        self.W_in = self.build_W_in(N_x, N_u, scaling_W_in)#input_W_in
        self.x = x_initial
        self.alpha_matrix = alpha_input

    def build_W(self, N_x, sparsity, scaling_W):
        # N_x integer
        # sparsity between 0 and 1 inclusive
        # scaling_W >= 1
        full_random = 2 * (np.random.rand(N_x, N_x) - 0.5)
        # Build sparse adjacency matrix W:
        input_W_sparsity = -1234 * np.ones((N_x, N_x))
        for i, row in enumerate(input_W_sparsity):
            for j, one_element in enumerate(row):
                randNum = np.random.rand()
                input_W_sparsity[i, j] = randNum < sparsity
        # Calculate unnormalized W
        W_unnormalized = np.multiply(input_W_sparsity, full_random)
        # Normalize by largest eigenvalue and additional scaling factor
        # to control decrease of spectral radius.
        largest_eigenvalue = np.sort(np.linalg.eigvals(W_unnormalized))[-1]
        W = np.multiply( scaling_W, np.divide(W_unnormalized,largest_eigenvalue) )
        return W

    def build_W_in(self, N_x, N_u, scaling_W_in):
        W_in = scaling_W_in*np.ones((N_x, N_u+1))
        return W_in

    def update_reservoir(self, u):
        # u is input at specific time
        #   u has shape (N_u + 1)
        x_nm1 = self.x
        x_n_tilde = np.tanh(np.matmul(self.W,x_nm1)
                            + np.matmul(self.W_in, np.hstack((u,np.array([1])))))
        x_n = np.mutiply((1-self.alpha_matrix), x_nm1) + \
              np.mutiply(self.alpha_matrix, x_n_tilde)
        self.x = x_n

    def output_Y(self, u):
        one_by_one = np.array([1])
        concatinated_matrix = np.hstack(one_by_one,
                                        u,
                                        self.x)
        return np.matmul(self.W_out, concatinated_matrix)

    def calculate_W_out(self, Y_target, X):
        # see Lukosevicius Practical ESN eqtn 11
        # Using the usual linear regression
        W_out = np.multi_dot(Y_target,
                             np.transpose(X),
                             np.linalg.inv(np.matmul(X,np.transpose(X))))
        self.W_out = W_out