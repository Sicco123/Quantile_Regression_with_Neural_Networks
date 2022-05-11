import tensorflow as tf

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


def optimize_neural_net(window, model, lambda_objective_function, max_deep_iter, learning_rate, patience = 5):

    #### Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience= patience,
                                                      mode='min')

    #####  Compile keras model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # default='rmsprop', an algorithm to be used in backpropagation
                  loss=lambda_objective_function, # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
                  steps_per_execution=1, # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
                  run_eagerly = False
                  )

    ##### Fit keras model on the dataset
    model.fit(window.train,  # target data
              epochs=max_deep_iter, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
              verbose=0, # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
              validation_data=window.val, # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch.
              shuffle=False,  # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
              callbacks = [early_stopping],
              workers=4,      # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
              use_multiprocessing=True,      # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False.
              )
    return model