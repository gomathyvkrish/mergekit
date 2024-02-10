import numpy as np
from scipy.optimize import linear_sum_assignment

# solve_lap IN Algorithm 1 while permuting both the rows and columns of W_A to match W_B
# Output of Hungarian Algorithm
# Matrix with ones at the coordinates indicated by the solution, with zeros everywhere else
def solve_lap(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # linear_sum_assignment maximizes the sum of elements given the cost matrix.
    perm_matrix = np.zeros_like(cost_matrix)
    perm_matrix[row_ind, col_ind] = 1
    return perm_matrix

# TODO Get weights of models A and B
W_A = [...]
W_B = [...]

L = len(W_A)  # Assuming both W_A and W_B have the same length L

# Initialize permutation matrices to identity
P = [np.eye(W_A[0].shape[0]) for _ in range(L - 1)]

# Repeat until convergence
converged = False
while not converged:
    # Random permutation of indices from 1 to L-1
    indices = np.random.permutation(range(1, L))

    for l in indices:
        # Calculate the cost matrix for the LAP using Frobenius inner product
        # Adding left and right cost matrix
        cost_matrix = W_A[l] @ P[l - 1] @ W_B[l].T
        if l < L - 1:  # If not the last layer, add the next term
            cost_matrix += W_A[l + 1].T @ P[l] @ W_B[l + 1]

        # Solve LAP and update the permutation matrix P_l
        P[l - 1] = solve_lap(cost_matrix)

    # TODO check convergence criteria
    converged = True

# Permutation matrices P is the result of the algorithm
