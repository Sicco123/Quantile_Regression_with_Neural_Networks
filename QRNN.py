import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Flatten  # to flatten the input shape
from l1_penalization import l1_p, non_cross_transformation


class QRNN_Dense(tf.keras.Model):
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

class QRNN_Conv(tf.keras.Model):
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

class QRNN_LSTM(tf.keras.Model):
    def __init__(self, hidden_dim,  output_dim, penalty_1 = 0, penalty_2 = 0):
        super().__init__()
        self.layer_1 = LSTM(hidden_dim, return_sequences = False, kernel_regularizer = tf.keras.regularizers.L2(penalty_2))
        self.layer_2 = Dropout(0.2)
        self.layer_3 = l1_p(number_of_quantiles=output_dim, activation="sigmoid", penalty_1 = penalty_1, penalty_2 = penalty_2)

        # possible add "Dropout" to prevent overfitting

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        self.pred, self.pred_mod = self.layer_3(x) # Output is a dictionary with the objective func input and intermediate results.
        return self.pred


# def objective_function(objective_y, input, quantiles):
#     predicted_y = input
#     print(objective_y, input)
#     ### prepare quantiles
#     quantile_length = len(quantiles)
#     quantile_tf = tf.convert_to_tensor(quantiles, dtype='float32')
#     quantile_tf_tiled = tf.repeat(tf.transpose(quantile_tf), [len(objective_y)])
#
#     ### prepare objective value
#     objective_y = tf.squeeze(tf.cast(objective_y, dtype = 'float32'))
#     output_y_tiled = tf.tile(objective_y, [quantile_length])
#
#     ### prepare predicted values
#     predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), [-1] ) #output_y_tiled.shape
#
#     ### objective function
#     diff_y = output_y_tiled - predicted_y_tiled
#     quantile_loss = tf.reduce_mean(diff_y * (quantile_tf_tiled - (tf.sign(-diff_y) + 1) / 2))
#
#     return quantile_loss
def quantile_risk(predicted_y, objective_y, quantiles):

    ### prepare quantiles
    quantile_length = len(quantiles)
    quantile_tf = tf.convert_to_tensor(quantiles, dtype='float32')
    quantile_tf_tiled = tf.repeat(tf.transpose(quantile_tf), [len(objective_y)])

    ### prepare objective value
    objective_y = tf.squeeze(tf.cast(objective_y, dtype='float32'))
    output_y_tiled = tf.tile(objective_y, [quantile_length])

    ### prepare predicted values
    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), [-1])  # output_y_tiled.shape

    ### objective function
    diff_y = output_y_tiled - predicted_y_tiled
    quantile_loss = tf.reduce_mean(diff_y * (quantile_tf_tiled - (tf.sign(-diff_y) + 1) / 2))
    return quantile_loss

def objective_function(objective_y_stack, input, quantiles):
     if len(objective_y_stack.shape) == 2:
         predicted_y = input
         objective_y = objective_y_stack
         ret_quantile_loss = quantile_risk(predicted_y, objective_y, quantiles)

     elif len(objective_y_stack.shape) == 3:
        window_length = objective_y_stack.shape[1]
        ret_quantile_loss = 0
        for idx in range(window_length):
             predicted_y = input[:, idx, :]
             objective_y = objective_y_stack[:, idx, :]
             quantile_loss = quantile_risk(predicted_y, objective_y, quantiles)

        ret_quantile_loss += quantile_loss

     return ret_quantile_loss


def optimize_l1_NMQN(window, model, lambda_objective_function, max_deep_iter, learning_rate,  penalty1=0, penalty2=0):

    #### Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5,
                                                      mode='min')

    #####  Compile keras model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # default='rmsprop', an algorithm to be used in backpropagation
                  loss=lambda_objective_function, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  #metrics=['Accuracy'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
                  steps_per_execution=1 # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                  )

    ##### Fit keras model on the dataset
    model.fit(window.train,  # target data
              epochs=max_deep_iter, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
              verbose=0, # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
              #validation_split=validation_split, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
              validation_data=window.val, # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
              shuffle=True,  # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
              # class_weight={0 : 0.3, 1 : 0.7}, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
              # sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
              callbacks = [early_stopping],
              initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
              steps_per_epoch=None,   # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
              workers=4,      # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
              use_multiprocessing=True,      # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
              )

    return model

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


def test_performance(df, model, split, window_length, quantiles, show_plot = False):
    weights = model.layers[-1].weights
    labels = df[split:].values[:,-1]
    store_feasible_output = []

    for t in range(len(df)-window_length):
        output = model.predict(df[t:t+window_length].values[np.newaxis])
        feasible_output = non_cross_transformation(output, weights[0], weights[1]).numpy()[0]
        feasible_output = np.squeeze(feasible_output)
        store_feasible_output.append(feasible_output)

    forecast = np.column_stack(store_feasible_output[split-window_length:])
    if show_plot:
        plot_results(labels, forecast, quantiles)

    print(f'The number of labels:{len(labels)}. The number of forecasts:{len(forecast[0])}.')

    quantile_loss = quantile_risk(forecast, labels, quantiles).numpy()
    print(quantile_loss)
    return quantile_loss


