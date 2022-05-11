import numpy as np
import pandas as pd
from arch.univariate import arch_model

def estimate_garch_quantiles(model, y, tau_vec, h, bootstrap_choices):
    residuals = model.resid[1:].values

    conditional_volatility = model.conditional_volatility[1:].values

    nlized_resid = residuals / np.sqrt(conditional_volatility)
    time = np.arange(1,len(nlized_resid),1)


    bootstrap_resid = nlized_resid[bootstrap_choices]

    phi0 = model.params['Const']
    phi1 = model.params['gdp[1]']
    omega = model.params['omega']
    alpha = model.params['alpha[1]']
    beta = model.params['beta[1]']

    mu_fut = phi0 + phi1 * y[-1]
    sigma_2_fut = model.conditional_volatility.values[-1]

    y_bootstrap = []

    for j in range(h):
            vol_fut = omega + alpha*np.square(y[-1]-mu_fut) + beta*sigma_2_fut
            mu_fut = phi0 + phi1 * y[-1]
            y_fut = mu_fut + bootstrap_resid/np.sqrt(vol_fut)
            y_bootstrap.append(y_fut)

    y_bootstrap = np.append(model.forecast(reindex = False).variance, y_bootstrap)
    quantiles = np.quantile(y_bootstrap, tau_vec)
    return quantiles

def GARCH_precidtions(train_df, val_df, test_df, df, tau_vec, h, bootstrap_choices):
    train_horizon = len(pd.concat([train_df,val_df], axis = 0))
    test_results = []

    for t in range(0,len(test_df)):
        model = arch_model(df.loc[t:train_horizon+t,'gdp'], mean = 'AR', lags= 1)
        res = model.fit(disp='off')

        predicted_quantiles = estimate_garch_quantiles(res,df['gdp'].values[:train_horizon+t+1],tau_vec,  h, bootstrap_choices[:,t])
        test_results.append(predicted_quantiles)

    return np.column_stack(test_results)