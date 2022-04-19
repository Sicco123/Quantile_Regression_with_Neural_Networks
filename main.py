import tensorflow as tf
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Flatten  # to flatten the input shape
import numpy as np
import matplotlib.pyplot as plt
from data_simulation import moon_data
from l1_penalization import l1_p

class QRNN(tf.keras.Model):
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

def objective_function(objective_y, input, quantiles):
    predicted_y = input

    ### prepare quantiles
    quantile_length = len(quantiles)
    quantile_tf = tf.convert_to_tensor(quantiles, dtype='float32')
    quantile_tf_tiled = tf.repeat(tf.transpose(quantile_tf), [len(objective_y)])

    ### prepare objective value
    objective_y = tf.squeeze(tf.cast(objective_y, dtype = 'float32'))
    output_y_tiled = tf.tile(objective_y, [quantile_length])

    ### prepare predicted values
    predicted_y_tiled = tf.reshape(tf.transpose(predicted_y), output_y_tiled.shape)

    ### objective function
    diff_y = output_y_tiled - predicted_y_tiled
    quantile_loss = tf.reduce_mean(diff_y * (quantile_tf_tiled - (tf.sign(-diff_y) + 1) / 2))

    return quantile_loss

def optimize_l1_NMQN_RMSProp(X_train, y_train, model, lambda_objective_function, learning_rate,  max_deep_iter, penalty1=0, penalty2=0):
    #####  Compile keras model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),  # default='rmsprop', an algorithm to be used in backpropagation
                  loss=lambda_objective_function, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  metrics=['Accuracy'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
                  steps_per_execution=5 # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                  )

    ##### Fit keras model on the dataset
    model.fit(X_train,  # input data
              y_train,  # target data
              batch_size=len(y_train), # Number of samples per gradient update. If unspecified, batch_size will default to 32.
              epochs=max_deep_iter, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
              verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
              validation_split=0, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
              # validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
              shuffle=False,  # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
              # class_weight={0 : 0.3, 1 : 0.7}, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
              # sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
              initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
              steps_per_epoch=None,   # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
              workers=4,      # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
              use_multiprocessing=True,      # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
              )

    return model

def main():
    ### Magic numbers
    n = 2000    # number of observations
    train_test_frac = 0.8   # train test split fraction
    input_dim = 1
    learning_rate = 0.005
    penalty_1 = 5 # l1 penalty of the delta coefficients
    penalty_2 = 0.0 # l2 penalty coef
    epochs = 2000

    tau_vec = np.arange(0.1, 1, 0.1)
    hidden_dim_1 = 4
    hidden_dim_2 = 4

    x, y = moon_data(n, input_dim)
    split_index = int(train_test_frac*n)
    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]

    ### Model fitting
    nmqn = QRNN(hidden_dim_1, hidden_dim_2, len(tau_vec), penalty_1, penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    nmqn_res = optimize_l1_NMQN_RMSProp(x_train, y_train, nmqn, loss_fn, learning_rate, epochs, penalty_1, penalty_2)

    print("Evaluate")
    res = nmqn_res(x_test)
    y_modified = nmqn_res.pred_mod

    plt.scatter(x_test, y_test)
    for i in range(len(tau_vec)):
        plt.scatter(x_test, y_modified[:,i])
    plt.show()


if __name__ == "__main__":
    main()