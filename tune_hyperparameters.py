from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import pandas as pd
from QRNN import QRNN_Dense, QRNN_Conv, QRNN_LSTM, objective_function,optimize_l1_NMQN
from window_data import WindowGenerator


def build_model_Dense(hp):
    units = hp.Int("units", min_value=4, max_value=32, step=4)
    lr = hp.Float("lr", min_value=1e-3, max_value=1e-2, step = 1e-3)
    penalty_1 = hp.Int("penalty_1", min_value=30, max_value=70, step = 10)
    penalty_2 = hp.Float("penalty_2", min_value = 0.000, max_value = 0.005, step = 0.001)
    # call existing model-building code with the hyperparameter values.
    tau_vec = np.append(0.05, np.arange(0.1, 1, 0.1))
    tau_vec = np.append(tau_vec, 0.95)
    model = QRNN_Dense(units, units, len(tau_vec), penalty_1 = penalty_1, penalty_2 = penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss_fn)
    return model

def build_model_Conv(hp):
    units = hp.Int("units", min_value=4, max_value=32, step=4)
    lr = hp.Float("lr", min_value=1e-3, max_value=1e-2, step = 1e-3)
    penalty_1 = hp.Int("penalty_1", min_value=30, max_value=70, step = 10)
    penalty_2 = hp.Float("penalty_2", min_value = 0.000, max_value = 0.005, step = 0.001)
    # call existing model-building code with the hyperparameter values.
    tau_vec = np.append(0.05, np.arange(0.1, 1, 0.1))
    tau_vec = np.append(tau_vec, 0.95)
    model = QRNN_Conv(units, units, len(tau_vec), kernel_size=4, penalty_1 = penalty_1, penalty_2 = penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss_fn)
    return model

def build_model_LSTM(hp):
    units = hp.Int("units", min_value=4, max_value=32, step=4)
    lr = hp.Float("lr", min_value=1e-3, max_value=1e-2, step = 1e-3)
    penalty_1 = hp.Int("penalty_1", min_value=30, max_value=70, step = 10)
    penalty_2 = hp.Float("penalty_2", min_value = 0.000, max_value = 0.005, step = 0.001)
    # call existing model-building code with the hyperparameter values.
    tau_vec = np.append(0.05, np.arange(0.1, 1, 0.1))
    tau_vec = np.append(tau_vec, 0.95)
    model = QRNN_LSTM(units, len(tau_vec), penalty_1 = penalty_1, penalty_2 = penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss_fn)
    return model


### Load Data
train_frac = 0.7   # train test split fraction
val_frac = 0.2
test_frac = 0.1

df = pd.read_csv('empirical_data/data_per_country/JPN.csv', index_col = 0)
n = len(df)
train_df = df[0:int(n * train_frac)]
val_df = df[int(n * train_frac):int(n * (train_frac + val_frac))]
split = int(n * (train_frac + val_frac))
test_df = df[int(n * (train_frac + val_frac)):]

model = build_model_Conv(kt.HyperParameters())
tuner = kt.Hyperband(
    build_model_Conv,
    objective='val_loss',
    max_epochs=100,
    directory = 'tuning',
    project_name = 'qrnn_conv')

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print ('ZZZZ\n')
tuner.search_space_summary()

wide_window_length = 4
window = WindowGenerator(
        input_width=wide_window_length, label_width=1, shift=1,
        train_df = train_df, val_df = val_df, test_df = test_df,
        label_columns=['gdp'])


tuner.search(window.train, epochs=500, validation_data= window.val , callbacks=[stop_early])
best_model = tuner.get_best_models()[0]
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)

print ('TTTT\n')
tuner.results_summary()

with open('parm.txt','w') as ff:
  ff.write(str(tuner.results_summary()))




