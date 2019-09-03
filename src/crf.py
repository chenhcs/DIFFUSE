import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

class CRF(object):
    def __init__(self, training_size, testing_size, positive_unary_energy, co_exp_net, theta, bag_label, bag_index):
        self.nodes_number = training_size + testing_size
        self.unary_potential = np.vstack((1 - positive_unary_energy, positive_unary_energy))
        self.co_exp_net = co_exp_net
        self.current_q = np.zeros((2, training_size + testing_size))
        self.negative_energy = np.zeros((2, training_size + testing_size))
        self.theta = theta
        self.bag_label = bag_label
        self.bag_index = bag_index
        self.label_update = np.zeros(training_size)
        self.pairwise_potential = np.zeros((2, training_size + testing_size))
        self.positive_ratio = bag_label[np.where(bag_label == 1)].shape[0] * 1.0 / bag_label[np.where(bag_label == 0)].shape[0]
        self.training_number = training_size
        self.testing_number = testing_size

    def parameter_learning(self, label, theta, sigma):
        def objective(theta, label, unary_potential, pairwise_potential, sigma):
            neg_energy = -unary_potential[:, 0:self.training_number] * theta[0] - pairwise_potential[:, 0:self.training_number] * theta[1]
            mx = np.max(neg_energy, 0)
            z = np.sum(np.exp(neg_energy - mx), 0)
            q = np.exp(neg_energy - mx) / z

            # print '______'
            # print 'Theta:', theta
            positive_index = np.where(label[0: self.training_number] == 1)
            negative_index = np.where(label[0: self.training_number] == 0)
            positive_number = len(positive_index[0])
            negative_number = len(negative_index[0])
            positive_rate = 0.1
            negative_rate = 1.0
            # print 'Cost:', -np.sum(np.log(q[label[positive_index[0]], positive_index[0]])) * negative_rate  - np.sum(np.log(q[label[negative_index[0]], negative_index[0]])) * positive_rate + np.sum(theta**2) / (2 * sigma)
            return -np.sum(np.log(q[label[positive_index[0]], positive_index[0]])) * negative_rate  - np.sum(np.log(q[label[negative_index[0]], negative_index[0]])) * positive_rate + np.sum(theta**2) / (2 * sigma)

        def gradient(theta, label, unary_potential, pairwise_potential, sigma):
            grad = [0, 0]

            neg_energy = -unary_potential[:, 0:self.training_number] * theta[0] - pairwise_potential[:, 0:self.training_number] * theta[1]
            mx = np.max(neg_energy, 0)
            z = np.sum(np.exp(neg_energy - mx), 0)
            q = np.exp(neg_energy - mx) / z
            # print q

            positive_index = np.where(label[0: self.training_number] == 1)
            negative_index = np.where(label[0: self.training_number] == 0)
            positive_number = len(positive_index[0])
            negative_number = len(negative_index[0])
            positive_rate = 0.1
            negative_rate = 1.0

            grad[0] = (np.sum(unary_potential[label[positive_index[0]], positive_index[0]]) - \
            (np.sum(q[0, positive_index[0]] * unary_potential[0, positive_index[0]]) + \
            np.sum(q[1, positive_index[0]] * unary_potential[1, positive_index[0]]))) * negative_rate + \
            (np.sum(unary_potential[label[negative_index[0]], negative_index[0]]) - \
            (np.sum(q[0, negative_index[0]] * unary_potential[0, negative_index[0]]) + \
            np.sum(q[1, negative_index[0]] * unary_potential[1, negative_index[0]]))) * positive_rate + \
            theta[0] / sigma

            grad[1] = (np.sum(pairwise_potential[label[positive_index[0]], positive_index[0]]) - \
            (np.sum(q[0, positive_index[0]] * pairwise_potential[0, positive_index[0]]) + \
            np.sum(q[1, positive_index[0]] * pairwise_potential[1, positive_index[0]]))) * negative_rate + \
            (np.sum(pairwise_potential[label[negative_index[0]], negative_index[0]]) - \
            (np.sum(q[0, negative_index[0]] * pairwise_potential[0, negative_index[0]]) + \
            np.sum(q[1, negative_index[0]] * pairwise_potential[1, negative_index[0]]))) * positive_rate + \
            theta[1] / sigma

            # print 'Grad:', grad
            return np.array(grad)

        opt_theta = fmin_l_bfgs_b(objective, theta, fprime=gradient, args=[label, self.unary_potential, self.pairwise_potential, sigma], epsilon=1e-08)
        return opt_theta[0]

    def inference(self, n_iterations, relax = 1.0):
        label_update = self.run_inference(n_iterations, relax)
        return label_update, self.current_q[1, :], self.unary_potential, self.pairwise_potential

    def run_inference(self, n_iterations, relax):
        self.exp_and_normalize(-1 * self.unary_potential, 0, 1, relax)
        for it in range(n_iterations):
            print 'Iteration:', it
            if it == n_iterations - 1:
                self.step_inference(relax, 1)
            else:
                self.step_inference(relax, 0)
        return self.label_update

    def step_inference(self, relax, label_gen):
        self.negative_energy = - self.theta[0] * self.unary_potential
        self.massage_passing()
        self.exp_and_normalize(self.negative_energy, label_gen, 1, relax)

    def massage_passing(self):
        product_mat = np.transpose(np.dot(self.co_exp_net, np.transpose(self.current_q)))
        self.pairwise_potential[0, :] = product_mat[1, :] / (2 * np.mean(product_mat))
        self.pairwise_potential[1, :] = product_mat[0, :] / (2 * np.mean(product_mat))
        self.negative_energy -= self.theta[1] * self.pairwise_potential

    def exp_and_normalize(self, neg_energy, label_gen, scale = 1.0, relax = 1.0):
        mx = np.max(scale * neg_energy, 0)
        z = np.sum(np.exp(scale * neg_energy - mx), 0)
        current_q = np.exp(scale * neg_energy - mx) / z
        self.current_q = (1 - relax) * self.current_q + relax * current_q

        if label_gen:
            self.label_update = np.argmax(self.current_q[:, 0: self.training_number], 0)
            for i in range(int(np.max(self.bag_index))):
                indx = np.where(self.bag_index == i)[0]
                if indx.shape[0] == 0:
                    continue
                bag_label = np.max(self.bag_label[indx])
                if bag_label == 1 and sum(self.label_update[indx]) == 0:
                    select = np.argmax(self.current_q[1, indx])
                    self.label_update[indx[select]] = 1
                elif bag_label == 0:
                    self.label_update[indx] = 0
