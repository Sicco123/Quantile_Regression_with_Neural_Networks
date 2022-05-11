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
from data_simulation import simulate_gaussian_ar_garch

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

def simulate_and_store(M, n, quantiles, garch_params):
    data = []
    for i in range(M):
        output, true_quantiles = simulate_gaussian_ar_garch(n, quantiles, **garch_params)
        if i == 0:
            data = true_quantiles
        else:
            data = np.column_stack((data, true_quantiles))
    np.savetxt('simulation_data/ar_garch_sim_data.csv',data, delimiter = ",")
    return

def main():
    ### Magic numbers
    epochs = 500
    h = 1
    B = 9999 # bootstrap replications
    M = 1000 # simulations
    n = 176    # number of observations
    train_frac = 0.7   # train test split fraction
    val_frac = 0.2
    test_frac = 0.1
    tau_vec = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    garch_params = {"const": 0.45, "ar_coef":0.40, "omega": 0.03, "garch_p_coef":0.13, "garch_q_coef":0.82}
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    bootstrap_choices = np.random.choice(158 - h, [B, 18], replace=True)
    simulate_data = False

    if simulate_data:
        simulate_and_store(M, n, tau_vec, garch_params)
    df = pd.read_csv('simulation_data/ar_garch_sim_data.csv', header = None)
    flat_data = df.values
    data = np.array([flat_data[:,(i*len(tau_vec)):(i*len(tau_vec)+len(tau_vec))] for i in range(M)])

    for i in range(M):
        print(f'Simulation data {i}/{M}')
        ### Load Data
        df_i = data[:,5,i]
        train_df, val_df, test_df = train_val_test_split(df_i, train_frac, val_frac)
        split = len(train_df) + len(val_df)

        ### GARCH Model
        forecast = GARCH_precidtions(train_df, val_df, test_df, df, tau_vec, h, bootstrap_choices)
        np.savetxt(f"simulation_experiment/GARCH/estimated_quantiles_GARCH_{i}.csv", forecast,
                   delimiter=",")

        ### DENSE QRNN
        hyper_params = [20, 0.007, 40, 0]  # units, learning_rate, p1, p2
        forecast = dense_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"simulation_experiment/Dense/estimated_quantiles_Dense_{i}.csv", forecast,
                   delimiter=",")

        ### Conv QRNN
        hyper_params = [12, 0.01, 60, 0]  # units, learning_rate, p1, p2
        forecast = conv_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"simulation_experiment/Conv/estimated_quantiles_Conv_{i}.csv", forecast, delimiter=",")

        ### LSTM QRNN
        hyper_params = [32, 0.008, 30, 0]  # units, learning_rate, p1, p2
        forecast = lstm_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"simulation_experiment/LSTM/estimated_quantiles_LSTM{i}.csv", forecast, delimiter=",")

        ### DENSE QRNN Adaptive learning rate
        lr_ini = 0.016
        decay = 0.5

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_ini,
            decay_steps=600,
            decay_rate=decay)

        hyper_params = [28, lr_schedule, 70, 0]  # units, learning_rate, p1, p2
        forecast = dense_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"simulation_experiment/Dense_adap/estimated_quantiles_Dense_adap_{i}.csv", forecast,
                   delimiter=",")

        ### Conv QRNN
        lr_ini = 0.02
        decay = 0.3

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_ini,
            decay_steps=600,
            decay_rate=decay)

        hyper_params = [8, lr_schedule, 40, 0]  # units, learning_rate, p1, p2
        forecast = conv_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"simulation_experiment/Conv_adap/estimated_quantiles_Conv_adap_{i}.csv", forecast,
                   delimiter=",")

        ### LSTM QRNN
        lr_ini = 0.016
        decay = 0.5

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_ini,
            decay_steps=600,
            decay_rate=decay)

        hyper_params = [12, lr_schedule, 70, 0]  # units, lr_schedule, p1, p2
        forecast = lstm_model(hyper_params, train_df, val_df, test_df, df, split, tau_vec, loss_fn, epochs)
        np.savetxt(f"simulation_experiment/LSTM_adap/estimated_quantiles_LSTM_adap_{i}.csv", forecast,
                   delimiter=",")


if __name__ == "__main__":
    main()