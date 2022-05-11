import numpy as np
import pandas as pd
import os
from tensorflow import keras
import tensorflow as tf
from window_data import WindowGenerator
from QRNN import DNN, CONV, LSTM, predictions
from GARCH import GARCH_precidtions
from tests import test_performance
from optimization import optimize_neural_net, objective_function

def train_val_test_split(df, train_frac, val_frac):
    n = len(df)
    train_df = df[0:int(n * train_frac)]
    val_df = df[int(n * train_frac):int(n * (train_frac + val_frac))]
    test_df = df[int(n * (train_frac + val_frac)):]
    return train_df, val_df, test_df

def dense_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs):
    # hyperparams
    units = hyper_params[0]
    lr = hyper_params[1]
    p1 = hyper_params[2]
    p2 = hyper_params[3]

    # set window
    window_length = 1
    window = WindowGenerator(input_width=window_length, label_width=1, shift=1,
                             train_df=train_df, val_df=val_df, test_df=test_df,
                             label_columns=['gdp'])
    # initialize model
    model = DNN(units, units, len(tau_vec), p1, p2)
    # fit model
    res = optimize_neural_net(window, model, loss_fn, epochs, lr, p1, p2)
    forecast = predictions(df, res, split, window_length)
    tf.keras.backend.clear_session()

    return forecast

def conv_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs):
    #hyperparams
    units = hyper_params[0]
    lr = hyper_params[1]
    p1 = hyper_params[2]
    p2 = hyper_params[3]

    #set window
    CONV_WIDTH = 4
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    # initialize model
    model = CONV(units, units, len(tau_vec), CONV_WIDTH, p1, p2)

    # fit model
    res = optimize_neural_net(conv_window, model, loss_fn, epochs, lr, p1, p2)

    # make predictions
    forecast =predictions(df, res, split, CONV_WIDTH)

    tf.keras.backend.clear_session()
    return forecast

def lstm_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs):
    # hyperparams
    units = hyper_params[0]
    lr = hyper_params[1]
    p1 = hyper_params[2]
    p2 = hyper_params[3]

    # set window
    wide_window_length = 4
    wide_window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    # initialize model
    model = LSTM(units, len(tau_vec), p1, p2)

    # fit model
    res = optimize_neural_net(wide_window, model, loss_fn, epochs, lr, p1, p2)

    forecast = predictions(df, res, split, wide_window_length)

    tf.keras.backend.clear_session()
    return forecast

def main():
    ### Magic numbers
    h = 1 # step ahead predicition
    train_frac = 0.7   # train test split fraction
    val_frac = 0.2
    test_frac = 0.1
    B = 999
    epochs = 2000
    tau_vec = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
    bootstrap_choices = np.random.choice(158-h, [B,18], replace=True)

    # Get the list of all files and directories
    path = "empirical_data/data_per_country"
    dir_list = os.listdir(path)
    dir_list.remove('plots_per_country')

    complete_results_qr = []
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)

    number_of_countries = len(dir_list)
    for idx, country_data in enumerate(dir_list):
        print(f'{country_data[:-4]}, {idx+1}/{number_of_countries}')
        ### Load Data
        df = pd.read_csv(f'empirical_data/data_per_country/{country_data}', index_col = 0)
        train_df, val_df, test_df = train_val_test_split(df, train_frac, val_frac)
        split = len(train_df) + len(val_df)

        ### GARCH Model
        forecast = GARCH_precidtions(train_df, val_df, test_df, df, tau_vec, h, bootstrap_choices)
        np.savetxt(f"estimated_quantiles/GARCH/estimated_quantiles_GARCH{country_data[:-4]}.csv", forecast,
                   delimiter=",")

        ### DENSE QRNN
        hyper_params = [20, 0.007, 40, 0] # units, learning_rate, p1, p2
        forecast = dense_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"estimated_quantiles/Dense/estimated_quantiles_Dense{country_data[:-4]}.csv", forecast, delimiter=",")

        ### Conv QRNN
        hyper_params = [12, 0.01, 60, 0] # units, learning_rate, p1, p2
        forecast = conv_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"estimated_quantiles/Conv/estimated_quantiles_Conv{country_data[:-4]}.csv", forecast, delimiter=",")

        ### LSTM QRNN
        hyper_params = [32, 0.008, 30, 0] # units, learning_rate, p1, p2
        forecast = lstm_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"estimated_quantiles/LSTM/estimated_quantiles_LSTM{country_data[:-4]}.csv", forecast, delimiter=",")

        ### DENSE QRNN Adaptive learning rate
        lr_ini = 0.016
        decay = 0.5

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_ini,
            decay_steps=600,
            decay_rate=decay)

        hyper_params = [28, lr_schedule, 70 ,0] # units, learning_rate, p1, p2
        forecast = dense_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"estimated_quantiles/Dense_adap/estimated_quantiles_Dense_adap_{country_data[:-4]}.csv", forecast, delimiter=",")

        ### Conv QRNN
        lr_ini = 0.02
        decay = 0.3

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_ini,
            decay_steps=600,
            decay_rate=decay)

        hyper_params = [8, lr_schedule, 40, 0]  # units, learning_rate, p1, p2
        forecast = conv_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"estimated_quantiles/Conv_adap/estimated_quantiles_Conv_adap_{country_data[:-4]}.csv", forecast, delimiter=",")

        ### LSTM QRNN
        lr_ini = 0.016
        decay = 0.5


        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_ini,
            decay_steps=600,
            decay_rate=decay)

        hyper_params = [12, lr_schedule, 70, 0] #units, lr_schedule, p1, p2
        forecast = lstm_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"estimated_quantiles/LSTM_adap/estimated_quantiles_LSTM_adap_{country_data[:-4]}.csv", forecast, delimiter=",")


if __name__ == "__main__":
    main()