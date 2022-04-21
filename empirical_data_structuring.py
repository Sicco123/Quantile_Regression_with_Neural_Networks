import pandas as pd
import numpy as np
source = "empirical_data"
save_location = "data_per_country"
start = "1973-Q1"
end = "2016-Q4"
start_date = "1973-01-01"
end_date = "2016-10-01"
data_gdp = pd.read_csv(source+"/gdp.csv", index_col = 0)
data_gdp = data_gdp.loc[data_gdp.index[data_gdp['Dates'] == start][0]:data_gdp.index[data_gdp['Dates'] == end][0], :]

data_nfci = pd.read_csv(source+"/nfci.csv", index_col = 0)
data_ts = pd.read_csv(source+"/ts.csv", index_col = 0)
data_ts = data_ts.loc[data_ts.index[data_ts['Dates'] == start_date][0]:data_ts.index[data_ts['Dates'] == end_date][0], :]
data_cs = pd.read_csv(source+"/cs.csv", index_col = 0)
data_cs = data_cs.loc[0:data_cs.index[data_cs['dates'] == end_date][0], :]
data_hp = pd.read_csv(source+"/hp.csv", index_col = 0)
data_hp = data_hp.loc[data_hp.index[data_hp['Dates'] == start_date][0]:data_hp.index[data_hp['Dates'] == end_date][0], :]
data_sv = pd.read_csv(source+"/sv.csv", index_col = 0)
data_sv = data_sv.loc[data_sv.index[data_sv['dates'] == start_date][0]:data_sv.index[data_sv['dates'] == end_date][0], :]


for country in data_gdp.columns[1:]:
    df_country = pd.DataFrame([])
    ### Exog. Variables
    #print(len(data_nfci[country]), len(data_ts[country]), len(data_hp[country]), len(data_sv['ret']), len(data_cs['ret']))
    df_country['nfci'] = data_nfci[country].reset_index(drop=True)
    df_country['ts'] = data_ts[country].reset_index(drop=True)
    df_country['hp'] = data_hp[country].reset_index(drop=True)
    df_country['sv'] = data_sv['ret'].reset_index(drop=True)
    #df_country['cs'] = data_cs['ret'].reset_index(drop = True)

    ### Endog. Objective Variable
    df_country['gdp'] = data_gdp[country].reset_index(drop = True)

    ### Store to csv
    df_country.to_csv(path_or_buf = source + "/" + save_location+"/"+country+".csv")


