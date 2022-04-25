from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
from QRNN import QRNN, objective_function,optimize_l1_NMQN

def place_holder(units, activation, dropout, validation , lr, penalty_1 , penalty_2):
    tau_vec = np.arange(0.1, 1, 0.1)
    model = QRNN(units, units, len(tau_vec), penalty_1 = penalty_1, penalty_2 = penalty_2)
    loss_fn = lambda x, z: objective_function(x, z, tau_vec)
    model = optimize_l1_NMQN(x_train, y_train, model, loss_fn, 1000, validation, lr, penalty_1,
                                penalty_2)

    return model


def build_model(hp):
    units = hp.Int("units", min_value=4, max_value=32, step=4)
    activation = hp.Choice("activation", ["sigmoid", "tanh"])
    dropout = hp.Boolean("dropout")
    validation = hp.Float("val_split", min_value=0.05, max_value= 0.4)
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    penalty_1 = hp.Int("penalty_1", min_value=5, max_value=100)
    penalty_2 = hp.Float("penalty_2", min_value = 0.000, max_value = 0.005)
    # call existing model-building code with the hyperparameter values.
    model = place_holder(
        units=units, activation=activation, dropout=dropout, validation = validation, lr=lr, penalty_1=penalty_1, penalty_2 = penalty_2
    )
    return model


train_test_frac = 0.8  # train validation split fraction
h = 1
data = np.genfromtxt('empirical_data/data_per_country/JPN.csv', delimiter = ',', skip_header = True)
X, y = data[:-h,:], data[h:,-1]
split_index = int(train_test_frac*len(y))
x_train, y_train = X[:split_index,:], y[:split_index]
x_test, y_test = X[split_index:], y[split_index:]

model = build_model(kt.HyperParameters())
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=500)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print ('ZZZZ\n')
tuner.search_space_summary()


tuner.search(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[stop_early])
best_model = tuner.get_best_models()[0]
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=100)[0]
print(best_hps)

print ('TTTT\n')
tuner.results_summary()

with open('parm.txt','w') as ff:
  ff.write(str(tuner.results_summary()))

best_model.summary()
pred_res = best_model.predict(x_test).flatten()

