import numpy as np

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
    def __init__(self, N_x, input_W_in, input_W_out,
                 x_initial, alpha_input):
        # num_rows and num_cols are scalars
        # input_W_in has shape (N_x, N_u + 1)
        # x_initial has shape (N_x, N_x)
        # alpha_input has shape (N_x, N_x)
        self.W = 2*(np.zeros((N_x, N_x))-1)
        self.W_in = input_W_in
        self.x = x_initial
        self.alpha_matrix = alpha_input
        self.W_out = input_W_out
    def update_reservoir(self, u):
        # u is input at specific time
        #   u has shape (N_u + 1)
        x_nm1 = self.x
        x_n_tilde = np.tanh(np.matmul(self.W,x_nm1)
                            + np.matmul(self.W_in, u))
        x_n = np.dot((1-self.alpha_matrix), x_nm1) + \
              np.dot(self.alpha_matrix, x_n_tilde)
        self.x = x_n
    def output(self, u):
        one_by_one = np.array([1])
        concatinated_matrix = np.hstack(one_by_one,
                                        u,
                                        self.x)
        return np.matmul(self.W_out, concatinated_matrix)
