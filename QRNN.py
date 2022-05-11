import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Flatten  # to flatten the input shape
from l1_penalization import l1_p, non_cross_transformation

class DNN(tf.keras.Model):
    def __init__(self, hidden_dim_1, hidden_dim_2, output_dim, penalty_1 = 0, penalty_2 = 0):
        super().__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(units=hidden_dim_1, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L2(penalty_2))  # Choose correct number of units
        self.layer_3 = Dense(units=hidden_dim_2, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_4 = l1_p(number_of_quantiles=output_dim, activation="sigmoid", penalty_1 = penalty_1, penalty_2 = penalty_2)
        # possible add "Dropout" to prevent overfitting

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        self.pred, self.pred_mod = self.layer_4(x) # Output is a dictionary with the objective func input and intermediate results.
        return self.pred

class CONV(tf.keras.Model):
    def __init__(self, hidden_dim_1, hidden_dim_2, output_dim, kernel_size,  penalty_1 = 0, penalty_2 = 0):
        super().__init__()
        self.layer_1 = Conv1D(filters=hidden_dim_1, kernel_size = (kernel_size,), activation = "sigmoid", kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_2 = Dense(units=hidden_dim_2, activation="sigmoid", kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_3 = l1_p(number_of_quantiles=output_dim, activation="sigmoid", penalty_1 = penalty_1, penalty_2 = penalty_2)
        # possible add "Dropout" to prevent overfitting

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x) #x
        self.pred, self.pred_mod = self.layer_3(x) # Output is a dictionary with the objective func input and intermediate results.
        return self.pred

class LSTM(tf.keras.Model):
    def __init__(self, hidden_dim,  output_dim, penalty_1 = 0, penalty_2 = 0):
        super().__init__()
        self.layer_1 = LSTM(hidden_dim, return_sequences = False, kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_2 = Dropout(0.0)
        self.layer_3 = l1_p(number_of_quantiles=output_dim, activation="sigmoid", penalty_1 = penalty_1, penalty_2 = penalty_2)

        # possible add "Dropout" to prevent overfitting

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        self.pred, self.pred_mod = self.layer_3(x) # Output is a dictionary with the objective func input and intermediate results.
        return self.pred


def predictions(df, model, split, window_length):

    weights = model.layers[-1].weights
    labels = df[split:].values[:, -1]
    store_feasible_output = []

    for t in range(len(df) - window_length):
        output = model.predict(df[t:t + window_length].values[np.newaxis])
        feasible_output = non_cross_transformation(output, weights[0], weights[1]).numpy()[0]
        feasible_output = np.squeeze(feasible_output)
        store_feasible_output.append(feasible_output)

    forecast = np.column_stack(store_feasible_output[split - window_length:])
    return forecast

def plot_results(labels, output, quantiles):
    predictions = output

    x = np.arange(0, len(labels), 1)
    plt.ylabel(f'gdp')
    for idx, quantile in enumerate(quantiles):
        plt.plot(x, predictions[idx,:], color=f'C{idx}', label=f'Q-{quantile:.2f}')

    plt.plot(x, labels, color='k', label='labels')
    plt.legend(ncol=int(len(quantiles) / 2 + 1), fontsize='x-small')
    plt.xlabel('Time [h]')
    plt.show()



