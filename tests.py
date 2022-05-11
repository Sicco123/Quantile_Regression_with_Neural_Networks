import numpy as np
import optimization
import l1_penalization

def back_testing(forecast, labels):
    violation_ratios = []
    for column in forecast:
        violation_ratios.append(len(column[np.where(labels > column)])/len(labels))
    return violation_ratios


def test_performance(df, model, split, window_length, quantiles, show_plot = False):
    weights = model.layers[-1].weights
    labels = df[split:].values[:,-1]
    store_feasible_output = []

    for t in range(len(df)-window_length):
        output = model.predict(df[t:t+window_length].values[np.newaxis])
        feasible_output = l1_penalization.non_cross_transformation(output, weights[0], weights[1]).numpy()[0]
        feasible_output = np.squeeze(feasible_output)
        store_feasible_output.append(feasible_output)

    forecast = np.column_stack(store_feasible_output[split-window_length:])
    #if show_plot:
    #    plot_results(labels, forecast, quantiles)

    quantile_loss = optimization.quantile_risk(forecast, labels, quantiles).numpy()
    violation_ratios = back_testing(forecast, labels)
    return quantile_loss, violation_ratios, store_feasible_output