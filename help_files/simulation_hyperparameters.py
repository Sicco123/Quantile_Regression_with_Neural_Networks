import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from window_data import WindowGenerator
from data_simulation import simulate_gaussian_ar_garch
import keras_tuner as kt
import tune_hyperparameters as th
import tensorflow as tf
import QRNN

def tune_DMM(train_df, val_df, test_df):
    model = th.build_model_Dense(kt.HyperParameters())
    tuner = kt.Hyperband(
        th.build_model_Dense,
        objective='val_loss',
        max_epochs=100,
        directory='tuning',
        project_name='qrnn_sim_dense')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print('DMM')
    print('ZZZZ\n')

    wide_window_length = 4
    window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    tuner.search(window.train, epochs=500, validation_data=window.val, callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models()[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(tuner.results_summary(1))
    return best_model


def tune_CONV(train_df, val_df, test_df):
    model = th.build_model_Conv(kt.HyperParameters())
    tuner = kt.Hyperband(
        th.build_model_Conv,
        objective='val_loss',
        max_epochs=100,
        directory='tuning',
        project_name='qrnn_sim_conv')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print('CONV')
    print('ZZZZ\n')

    wide_window_length = 4
    window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    tuner.search(window.train, epochs=500, validation_data=window.val, callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models()[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(tuner.results_summary(1))
    return best_model

def tune_LSTM(train_df, val_df, test_df):
    model = th.build_model_LSTM(kt.HyperParameters())
    tuner = kt.Hyperband(
        th.build_model_LSTM,
        objective='val_loss',
        max_epochs=100,
        directory='tuning',
        project_name='qrnn_sim_lstm')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print('LSTM')
    print('ZZZZ\n')

    wide_window_length = 4
    window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    tuner.search(window.train, epochs=500, validation_data=window.val, callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models()[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(tuner.results_summary(1))
    return best_model

def tune_DMM_adap(train_df, val_df, test_df):
    model = th.build_model_Dense_adap(kt.HyperParameters())
    tuner = kt.Hyperband(
        th.build_model_Dense_adap,
        objective='val_loss',
        max_epochs=100,
        directory='tuning',
        project_name='qrnn_sim_dense_adap')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print('DMM_adap')
    print('ZZZZ\n')

    wide_window_length = 1
    window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    tuner.search(window.train, epochs=500, validation_data=window.val, callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models()[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(tuner.results_summary(1))
    return best_model

def tune_CONV_adap(train_df, val_df, test_df):
    model = th.build_model_Conv_adap(kt.HyperParameters())
    tuner = kt.Hyperband(
        th.build_model_Conv_adap,
        objective='val_loss',
        max_epochs=100,
        directory='tuning',
        project_name='qrnn_sim_conv_adap')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print('CONV_adap')
    print('ZZZZ\n')

    wide_window_length = 4
    window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    tuner.search(window.train, epochs=500, validation_data=window.val, callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models()[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(tuner.results_summary(1))
    return best_model

def tune_LSTM_adap(train_df, val_df, test_df):
    model = th.build_model_LSTM_adap(kt.HyperParameters())
    tuner = kt.Hyperband(
        th.build_model_LSTM_adap,
        objective='val_loss',
        max_epochs=100,
        directory='tuning',
        project_name='qrnn_sim_lstm_adap')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print('LSTM_adap')
    print('ZZZZ\n')

    wide_window_length = 4
    window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['gdp'])

    tuner.search(window.train, epochs=500, validation_data=window.val, callbacks=[stop_early], verbose=0)
    best_model = tuner.get_best_models()[0]
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    print(tuner.results_summary(1))
    return best_model



def simulate_and_store(M, n, quantiles, garch_params):
    data = []
    for i in range(M):
        output, true_quantiles = simulate_gaussian_ar_garch(n, quantiles, **garch_params)
        if i == 0:
            data = true_quantiles
        else:
            data = np.column_stack((data, true_quantiles))
    np.savetxt('simulation_data/ar_garch_sim_data.csv',data, delimiter = ",")
    print(data.shape)
    return data
def main():
    ### Magic numbers
    M = 100 # simulations
    n = 176    # number of observations
    train_frac = 0.7   # train test split fraction
    val_frac = 0.2
    test_frac = 0.1
    tau_vec = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    garch_params = {"const": 0.45, "ar_coef":0.40, "omega": 0.03, "garch_p_coef":0.13, "garch_q_coef":0.82}

    df = pd.read_csv('../simulation_data/ar_garch_sim_data.csv', header = None)
    flat_data = df.values
    data = np.array([flat_data[:,(i*len(tau_vec)):(i*len(tau_vec)+len(tau_vec))] for i in range(M)])



    #### Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5,
                                                      mode='min')

    data_m = pd.DataFrame(data[0,:,5:6], columns = ['gdp'])
    print(data_m)
    train_df = data_m[0:int(n * train_frac)]
    val_df = data_m[int(n * train_frac):int(n * (train_frac + val_frac))]
    split = int(n * (train_frac + val_frac))
    test_df = data_m[int(n * (train_frac + val_frac)):]

    DMM_model = tune_DMM(train_df, val_df, test_df)
    CONV_model = tune_CONV(train_df, val_df, test_df)
    LSTM_model = tune_LSTM(train_df, val_df, test_df)
    DMM_adap_model = tune_DMM_adap(train_df, val_df, test_df)
    CONV_adap_model = tune_CONV_adap(train_df, val_df, test_df)
    LSTM_adap_model = tune_LSTM_adap(train_df, val_df, test_df)





if __name__ == "__main__":
    main()