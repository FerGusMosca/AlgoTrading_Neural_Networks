import time
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

from common.enums.on_off_indicator_values import OnOffIndicatorValue


class ARIMAModelsAnalyzer():
    def __init__(self, p_logger):
        self.logger = p_logger



    #region privste methods

    def __aggregate_weekly_basis__(self,df,symbol,period):

        if period is not None:
            df.set_index('date', inplace=True)
            df_period = df.resample(period).mean()
            df_period = df_period[[symbol]]
            return df_period
        else:
            return df[[symbol]]

    def __add_weekly_returns__(self, df_period, symbol):
        df_period = df_period.copy()  # Evita modificar el original por referencia
        df_period['period_ret'] = np.log(df_period[symbol]).diff()

        # drop null rows
        df_period.dropna(inplace=True)

        # Asegurar que el índice es un DatetimeIndex
        if not isinstance(df_period.index, pd.DatetimeIndex):
            df_period.index = pd.to_datetime(df_period.index)

        return df_period[['period_ret']]  # Devuelve el DataFrame con el índice original

    def __plot_rolling_mean_std_dev__(self, udiff):
        if isinstance(udiff, pd.DataFrame):
            udiff = udiff["period_ret"]

        if not isinstance(udiff.index, pd.DatetimeIndex):
            udiff.index = pd.date_range(start="2000-01-01", periods=len(udiff), freq="M")

        rolmean = udiff.rolling(5).mean()
        rolstd = udiff.rolling(5).std()

        plt.figure(figsize=(12, 6))
        plt.plot(udiff, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std Deviation')
        plt.title('Rolling Mean & Standard Deviation')
        plt.legend(loc='best')
        plt.show(block=True)

    #The ACF gives us a measure of how much each "y" value is correlated to the previous n "y" values prior.
    #Helps us choose the q parameter
    def __acf_auto_corr_plot__(self, udiff):
        if isinstance(udiff, pd.DataFrame):
            udiff = udiff["period_ret"]

        if not isinstance(udiff.index, pd.DatetimeIndex):
            udiff.index = pd.date_range(start="2000-01-01", periods=len(udiff), freq="M")

        fig, ax = plt.subplots(figsize=(12, 5))
        plot_acf(udiff.dropna(), lags=10, ax=ax)  # Evitar valores NaN en ACF
        plt.show(block=True)

    #he PACF is the partial correlation function gives us (a sample of) the amount of correlation between two "y" values
    # separated by n lags excluding the impact of all the "y" values in between them.
    # Helps us chose the p parameter
    def __pacf_auto_corr_plot__(self, udiff):
        if isinstance(udiff, pd.DataFrame):
            udiff = udiff["period_ret"]

        if not isinstance(udiff.index, pd.DatetimeIndex):
            udiff.index = pd.date_range(start="2000-01-01", periods=len(udiff), freq="M")

        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(udiff.dropna(), lags=10, ax=ax)  # Evitar valores NaN en PACF
        plt.show(block=True)

    def __perf_dickey_fuller_test__(self, udiff):
        # Realizar el test de Dickey-Fuller
        dftest = sm.tsa.adfuller(udiff.period_ret, autolag='AIC')

        # Extraer los resultados
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

        # Guardar valores críticos
        dickey_fuller_test_dict = {key: value for key, value in dftest[4].items()}

        # Incluir el estadístico en el output
        dickey_fuller_test_dict['Test Statistic'] = dftest[0]  # Agregar el estadístico de prueba
        dickey_fuller_test_dict['p-value'] = dftest[1]  # Agregar el p-valor

        print("\n======= Showing Dickey-Fuller Test after building ARIMA =======")
        for key, value in dickey_fuller_test_dict.items():
            print(f"{key} = {value}")

        return dickey_fuller_test_dict

    def __build_ARIMA__(self,udiff,p,d,q):
        # Notice that you have to use udiff - the differenced data rather than the original data.
        ar1 = ARIMA(udiff.values, order=(p,d,q)).fit()
        ar1.summary()
        return ar1



    def __run_forecast__(self,udiff,ar1,steps):

        forecast = ar1.forecast(steps=steps)
        preds = ar1.fittedvalues
        # plt.figure(figsize=(12, 8))
        # plt.plot(udiff.values, color='blue')
        # plt.plot(preds, color='red')
        #
        # plt.plot(pd.DataFrame(np.array([preds[-1], forecast[0]]).T,
        #                       index=range(len(udiff.values) + 1, len(udiff.values) + 3)), color='green')
        # plt.plot(pd.DataFrame(forecast, index=range(len(udiff.values) + 1, len(udiff.values) + 1 + steps)),
        #          color='green')
        # plt.title('Display the predictions with the ARIMA model')
        # plt.show()
        return forecast


    def __show_dick_fuller__(self,pred_dict):
        print("======= Showing Dickey Fuller Test after building ARIMA======= ")
        for key in pred_dict.keys():
            print("{}={}".format(key, pred_dict[key]))

    #endregion


    def build_ARIMA_model(self,series_df,symbol,period,show_graphs=True):
        df_period=self.__aggregate_weekly_basis__(series_df,symbol,period)
        udiff=self.__add_weekly_returns__(df_period,symbol)

        #Print Dickey Fuller on the Screen
        dickey_fuller_test_dict=self.__perf_dickey_fuller_test__(udiff)
        self.__show_dick_fuller__(dickey_fuller_test_dict)

        if show_graphs:
            self.__plot_rolling_mean_std_dev__(udiff)
            self.__acf_auto_corr_plot__(udiff)
            self.__pacf_auto_corr_plot__(udiff)



        return dickey_fuller_test_dict


    def build_and__predict_ARIMA_model(self,series_df,symbol,p,d,q,period,step):
        df_period=self.__aggregate_weekly_basis__(series_df,symbol,period)
        udiff=self.__add_weekly_returns__(df_period,symbol)
        ar1=self.__build_ARIMA__(udiff,p,d,q)

        preds=self.__run_forecast__(udiff,ar1,step)

        forecast_list = preds.tolist()

        return forecast_list


    def eval_still_on_indicator(self,preds,step,inv_steps):

        if inv_steps>step:
            raise (f"Cannot evaluate {inv_steps} steps when we only have {step} in the predictions ")

        loop_sign=None
        for i, pred in enumerate(preds):
            if i >= inv_steps:
                break  #Cut inv_steps
            loop_sign_curr=  OnOffIndicatorValue.ON_NUMERIC.value if pred>0 else  OnOffIndicatorValue.OFF_NUMERIC.value

            if loop_sign is not None and loop_sign!=loop_sign_curr:
                return loop_sign_curr

            loop_sign=loop_sign_curr


        return loop_sign ==  OnOffIndicatorValue.ON_NUMERIC.value




