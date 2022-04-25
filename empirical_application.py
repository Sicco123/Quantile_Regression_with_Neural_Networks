import tensorflow as tf
from keras.layers import Dense  # for creating regular densely-connected NN layers.
from keras.layers import Flatten  # to flatten the input shape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from window_data import WindowGenerator
from data_simulation import moon_data
from l1_penalization import l1_p
from QRNN import QRNN_Dense, QRNN_Conv, QRNN_LSTM, objective_function, optimize_l1_NMQN



def main():
    ### Magic numbers
    h = 1 # step ahead predicition
    train_frac = 0.7   # train test split fraction
    val_frac = 0.2
    test_frac = 0.1
    #validation_split = 0.22 # train validation split fraction
    learning_rate = 0.00218
    penalty_1 = 60 # l1 penalty of the delta coefficients
    penalty_2 = 0.00036 # l2 penalty coef
    epochs = 2000
    tau_vec = np.append(0.05, np.arange(0.1, 1, 0.1))
    tau_vec = np.append(tau_vec, 0.95)
    hidden_dim_1 = 12
    hidden_dim_2 = 12
    validation_results = []
    test_results = []

    ### Load Data
    df = pd.read_csv('empirical_data/data_per_country/JPN.csv', index_col = 0)
    n = len(df)
    train_df = df[0:int(n * train_frac)]
    val_df = df[int(n * train_frac):int(n * (train_frac + val_frac))]
    test_df = df[int(n * (train_frac + val_frac)):]

    # ### DENSE QRNN
    # window = WindowGenerator(input_width=4, label_width=1, shift=1,
    #                          train_df=train_df, val_df=val_df, test_df=test_df,
    #                          label_columns=['gdp'])
    # # Model fitting
    # nmqn = QRNN_Dense(hidden_dim_1, hidden_dim_2, len(tau_vec), penalty_1, penalty_2)
    # loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    # nmqn_res = optimize_l1_NMQN(window, nmqn, loss_fn, epochs, learning_rate, penalty_1, penalty_2)
    # val_loss = nmqn_res.evaluate(window.val)
    # test_loss = nmqn_res.evaluate(window.test)
    #
    # # store results
    # validation_results.append(val_loss)
    # test_results.append(test_loss)
    #
    #
    # ### Conv QRNN
    # CONV_WIDTH = 4
    # conv_window = WindowGenerator(
    #     input_width=CONV_WIDTH, label_width=1, shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['gdp'])
    #
    # # Model fitting
    # nmqn = QRNN_Conv(hidden_dim_1, hidden_dim_2, len(tau_vec), CONV_WIDTH, penalty_1, penalty_2)
    # loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    # nmqn_res = optimize_l1_NMQN(conv_window, nmqn, loss_fn, epochs, learning_rate, penalty_1, penalty_2)
    #
    # val_loss = nmqn_res.evaluate(window.val)
    # test_loss = nmqn_res.evaluate(window.test)
    #
    # # store results
    # validation_results.append(val_loss)
    # test_results.append(test_loss)

    ### LSTM QRNN
    wide_window = WindowGenerator(
        input_width=15, label_width=15, shift=1,
        train_df = train_df, val_df = val_df, test_df = test_df,
        label_columns=['gdp'])

    # Model fitting
    nmqn = QRNN_LSTM(hidden_dim_1, len(tau_vec), penalty_1, penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    nmqn_res = optimize_l1_NMQN(wide_window, nmqn, loss_fn, epochs, learning_rate, penalty_1, penalty_2)

    val_loss = nmqn_res.evaluate(wide_window.val)
    test_loss = nmqn_res.evaluate(wide_window.test)

    # store results
    validation_results.append(val_loss)
    test_results.append(test_loss)

    ### Print results
    print(validation_results)
    print(test_results)


    # LABEL_WIDTH = 16
    # INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    # wide_conv_window = WindowGenerator(
    #     input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['gdp'])
    #
    #
    # wide_window = WindowGenerator(
    #     input_width=15, label_width=15, shift=1,
    #     train_df = train_df, val_df = val_df, test_df = test_df,
    #     label_columns=['gdp'])


    # X, y = data[:-h,:], data[h:,-1]
    # split_index = int(train_test_frac*len(y))
    # x_train, y_train = X[:split_index,:], y[:split_index]
    # x_test, y_test = X[split_index:], y[split_index:]


    # LABEL_WIDTH = 24
    # INPUT_WIDTH = LABEL_WIDTH + (4 - 1)
    # wide_conv_window = WindowGenerator(
    #     input_width=INPUT_WIDTH,
    #     label_width=LABEL_WIDTH,
    #     shift=1,
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     label_columns=['gdp'])
    #
    # wide_window.plot(nmqn_res, tau_vec)
    #window.plot_test_result(nmqn_res, tau_vec)


    # print("Evaluate")
    # res = nmqn_res(window.test)
    # y_modified = nmqn_res.pred_mod
    #
    # time = np.arange(0,len(y_modified), 1)
    # plt.plot(time, test_df['gdp'])
    # for i in range(len(tau_vec)):
    #     plt.plot(time, y_modified[:,i])
    # plt.show()


if __name__ == "__main__":
    main()