import tensorflow as tf
from keras import layers
from keras import activations
import numpy as np


class l1_p(layers.Layer): #This class builds on the layer class of keras.
    def __init__(self, number_of_quantiles, activation, penalty_1, penalty_2, **kwargs):
        super(l1_p, self).__init__()
        self.activation = activations.get(activation)  # Choose an activation which is supported by keras
        self.number_of_quantiles = number_of_quantiles
        self.penalty_1 = penalty_1
        self.penalty_2 = penalty_2

    def build(self, input_shape):
        self.input_shape_1 = input_shape[-1]    # Equals the number of neurons in the previous layer

        ### Build delta
        self.delta_coef_matrix = tf.Variable( tf.random.normal( shape = [self.input_shape_1, self.number_of_quantiles]))
        self.delta_0_matrix = tf.Variable( tf.random.normal( shape=[1, self.number_of_quantiles]))


    def call(self, inputs, **kwargs):

        ### Build beta matrix
        delta_mat = tf.concat([self.delta_0_matrix, self.delta_coef_matrix], axis=0)
        beta_mat = tf.transpose(tf.cumsum(tf.transpose(delta_mat)))

        delta_vec = delta_mat[1:, 1:] # leave out the first column
        delta_0_vec = delta_mat[0:1, 1:]
        delta_minus_vec = tf.maximum(0.0, -delta_vec)
        delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, axis=0)
        delta_0_vec_clipped = tf.clip_by_value(delta_0_vec, # clip to ensure feasibility of delta_0_vec
                                               clip_value_min=tf.reshape(delta_minus_vec_sum, delta_0_vec.shape),
                                               clip_value_max=tf.convert_to_tensor(
                                                   (np.ones(np.shape(delta_0_vec)) * np.inf), dtype='float32'))

        predicted_y = tf.matmul(inputs, beta_mat[1:, :]) + beta_mat[0, :]
        predicted_y_modified = tf.matmul(inputs, beta_mat[1:, :]) + tf.cumsum(tf.concat([beta_mat[0:1, 0:1],
                                                                                              delta_0_vec_clipped],
                                                                                             axis=1),
                                                                                   axis=1)

        delta_constraint = delta_0_vec_clipped - delta_minus_vec_sum
        delta_clipped = tf.clip_by_value(delta_constraint, clip_value_min=10 ** (-20), clip_value_max=np.Inf)
        delta_l1_penalty = tf.reduce_mean(tf.abs(
            delta_0_vec - delta_0_vec_clipped))  # tf.reduce_mean(tf.abs(self.delta_0_vec - self.delta_0_vec_clipped))
        self.add_loss(self.penalty_1 * delta_l1_penalty)

        l2_penalty = tf.reduce_mean(self.delta_coef_matrix**2) # delta penalty
        self.add_loss(self.penalty_2 * l2_penalty)

        return predicted_y, predicted_y_modified

def non_cross_transformation(predicted_y, delta_coef_matrix, delta_0_matrix):
    ### Build beta matrix
    delta_mat = tf.concat([delta_0_matrix, delta_coef_matrix], axis=0)
    beta_mat = tf.transpose(tf.cumsum(tf.transpose(delta_mat)))

    delta_vec = delta_mat[1:, 1:]  # leave out the first column
    delta_0_vec = delta_mat[0:1, 1:]
    delta_minus_vec = tf.maximum(0.0, -delta_vec)
    delta_minus_vec_sum = tf.reduce_sum(delta_minus_vec, axis=0)
    delta_0_vec_clipped = tf.clip_by_value(delta_0_vec,  # clip to ensure feasibility of delta_0_vec
                                           clip_value_min=tf.reshape(delta_minus_vec_sum, delta_0_vec.shape),
                                           clip_value_max=tf.convert_to_tensor(
                                               (np.ones(np.shape(delta_0_vec)) * np.inf), dtype='float32'))

    part_1 = predicted_y -  beta_mat[0, :]
    transformed_y = part_1 + tf.cumsum(tf.concat([beta_mat[0:1, 0:1],
                                                 delta_0_vec_clipped], axis=1), axis=1)
    return transformed_y