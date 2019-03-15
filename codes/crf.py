import numpy as np
import time

class CRF(object):
    def __init__(self, testing_size, positive_unary_energy, co_exp_net, theta):
        self.unary_potential = np.vstack((1 - positive_unary_energy, positive_unary_energy))
        self.co_exp_net = co_exp_net
        self.current_q = np.zeros((2, testing_size))
        self.negative_energy = np.zeros((2, testing_size))
        self.theta = theta
        self.pairwise_potential = np.zeros((2, testing_size))

    def inference(self, n_iterations, relax = 1.0):
        self.run_inference(n_iterations, relax)
        return self.current_q[1, :]

    def run_inference(self, n_iterations, relax):
        self.exp_and_normalize(-1 * self.unary_potential, 1, relax)
        for it in range(n_iterations):
            print 'Iteration:', it
            self.step_inference(relax)

    def step_inference(self, relax):
        self.negative_energy = - self.theta[0] * self.unary_potential
        self.massage_passing()
        self.exp_and_normalize(self.negative_energy, 1, relax)

    def massage_passing(self):
        product_mat = np.transpose(np.dot(self.co_exp_net, np.transpose(self.current_q)))
        self.pairwise_potential[0, :] = product_mat[1, :] / (2 * np.mean(product_mat))
        self.pairwise_potential[1, :] = product_mat[0, :] / (2 * np.mean(product_mat))
        self.negative_energy -= self.theta[1] * self.pairwise_potential

    def exp_and_normalize(self, neg_energy, scale = 1.0, relax = 1.0):
        mx = np.max(scale * neg_energy, 0)
        z = np.sum(np.exp(scale * neg_energy - mx), 0)
        current_q = np.exp(scale * neg_energy - mx) / z
        self.current_q = (1 - relax) * self.current_q + relax * current_q
