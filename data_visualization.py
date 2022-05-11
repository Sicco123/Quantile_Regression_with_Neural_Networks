import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy import stats

def get_countries():
    path = "empirical_data/data_per_country"
    dir_list = os.listdir(path)
    dir_list.remove('plots_per_country')
    countries = [country_csv[:-4] for country_csv in dir_list]
    return countries

def plot_histogram(df):
    means = df.mean(axis=0).values
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    x = df.columns.values
    y = means

    ax.bar(x, y, color = ['C0','C1','C2','C3','C1','C2','C3'], alpha= 0.8)
    ax.set_ylabel('Quantile Risk', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Out of Sample Mean Quantile Risk')
    #for i, v in enumerate(y):
    #    ax.text(i, v + 1, str(round(v,2)), color='black', fontweight='bold')
    plt.savefig('plots/mean_quantile_risk.png', dpi=200, format='png', bbox_inches='tight')
    plt.show()
    plt.close()
def t_test(data):
    results = np.zeros((len(data.columns),len(data.columns)))
    for i, model_1 in enumerate(data.columns):
        for j, model_2 in enumerate(data.columns):
            if i == j:
                results[i,j] = 1
                continue
            test_res = stats.ttest_ind(data[model_1], data[model_2], equal_var = False, alternative = 'less')
            results[i,j] = test_res[1]

    results_df = pd.DataFrame(results, columns = data.columns, index = data.columns)
    print(results_df.to_latex(float_format="%.3f"))

def calculate_qrnn_mape(average_bt_res, tau_vec):
    expected = np.tile(tau_vec, [6,1]).T
    subtract = np.abs(average_bt_res - expected)
    division = np.divide(subtract, expected)*100
    average_mape = np.mean(division, axis=0)
    return average_mape

def calculate_garch_mape(bt_res, tau_vec):
    expected = np.tile(tau_vec, [24,1]).T
    subtract = np.abs(bt_res - expected)
    division = np.divide(subtract, expected)*100
    average_mape = np.mean(division, axis = 0)
    return average_mape

def main():
    tau_vec = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    results_qrnn_source = 'estimated_quantiles/complete_simulation_results_qr.csv'
    results_garch_source = 'emperical_results/garch_results.csv'
    model_names = ['Dense 1', 'Conv 1', 'LSTM 1', 'Dense 2', 'Conv 2', 'LSTM 2']
    countries = get_countries()

    # results_qrnn_qr = pd.read_csv(results_qrnn_source, header=None)
    # results_qrnn_qr.columns = model_names
    # results_qrnn_qr.index = countries
    #
    # res_garch = pd.read_csv(results_garch_source, header = None).T
    # res_garch.index = countries
    # res_garch.columns = ["Garch"]
    #
    # data = pd.concat([res_garch, results_qrnn_qr], axis=1)
    # plot_histogram(data)
    # t_test(data)

    results_qrnn_bt_root_source = "estimated_quantiles/back_test/"
    mape_values = []

    # Garch
    df = pd.read_csv(f"emperical_results/empirical_backtest_results_garch.csv", header = None)
    garch_mape = calculate_garch_mape(df, tau_vec)
    garch_mape = pd.DataFrame(garch_mape, columns = ['Garch'])

    for country in countries:
        df = pd.read_csv(f"{results_qrnn_bt_root_source}complete_simulation_results_bt_{country}.csv", header = None)
        df.columns = model_names
        average_mape = calculate_qrnn_mape(df, tau_vec)
        mape_values.append(average_mape)
    mape_values = pd.DataFrame(mape_values)
    mapes = pd.concat([garch_mape, mape_values], axis=1)
    plot_histogram(mapes)

if __name__ == "__main__":
    main()