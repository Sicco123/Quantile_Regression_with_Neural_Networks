import tensorflow as tf
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Flatten  # to flatten the input shape
import numpy as np
import matplotlib.pyplot as plt
from data_simulation import moon_data
from l1_penalization import l1_p
from optimization import  objective_function
from QRNN import DNN

def plot_quantiles(x_test, y_test, tau_vec, quantiles, save_name):
    plt.scatter(x_test, y_test)
    for i in range(len(tau_vec)):
        plt.scatter(x_test, quantiles[:, i])
    plt.savefig(save_name)
    plt.show()

def optimize_neural_net(X_train, y_train, model, lambda_objective_function, learning_rate,  max_deep_iter, penalty1=0, penalty2=0):
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

    #simulate data
    x, y, true_quantiles = moon_data(n, input_dim)
    split_index = int(train_test_frac*n)
    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]
    true_quantiles_test = true_quantiles[split_index:]

    ### Model fitting
    model = DNN(hidden_dim_1, hidden_dim_2, len(tau_vec), penalty_1, penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    model = optimize_neural_net(x_train, y_train, model, loss_fn, learning_rate, epochs)

    print("Evaluate")
    res = model(x_test)
    y_modified = model.pred_mod

    plot_quantiles(x_test, y_test, tau_vec, y_modified, 'moon_plots/estimation_res.png')
    plot_quantiles(x_test, y_test, tau_vec, true_quantiles_test, 'moon_plots/true_quantiles.png')

if __name__ == "__main__":
    main()