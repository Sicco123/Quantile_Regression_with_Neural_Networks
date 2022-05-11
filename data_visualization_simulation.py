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
    plt.savefig('plots/sim_one_ahead_garch_mean_quantile_risk.png', dpi=200, format='png', bbox_inches='tight')
    #plt.show()
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
    print(np.round(division,3))
    print(average_mape)

def main():
    tau_vec = np.array([0.95,0.9,0.8,0.7, 0.6,0.5,0.4,0.3,0.2,0.1,0.05])
    results_qrnn_source = 'simulation_experiment/complete_simulation_results_qr.csv'
    results_garch_source = 'simulation_experiment/simulation_results_GARCH.csv'
    model_names = ['Dense', 'Conv', 'LSTM']

    garch_qr_results = pd.read_csv(results_garch_source, header= None).T
    qrnn_qr_results = pd.read_csv(results_qrnn_source, header = None)

    qr_results = pd.DataFrame(np.column_stack((garch_qr_results, qrnn_qr_results)))
    qr_results.columns = ['Garch', 'Dense 1', 'Conv 1', 'LSTM 1', 'Dense 2', 'Conv 2', 'LSTM 2' ]


    plot_histogram(qr_results)
    t_test(qr_results)
    i = 0

    average_bt_res = np.zeros((11,6))
    for i in range(100):
        results_qrnn_bt_source = f'simulation_experiment/complete_simulation_results_bt_{i}.csv'
        results_qrnn_bt = pd.read_csv(results_qrnn_bt_source, header = None)
        average_bt_res = average_bt_res + results_qrnn_bt.values

    average_bt_res = average_bt_res/100
    QRNN_mape_results = calculate_qrnn_mape(average_bt_res, tau_vec)

if __name__ == "__main__":
    main()