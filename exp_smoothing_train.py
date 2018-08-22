import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import collections
import argparse
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


class Exp_Smoothing:
        def __init__(self, train, test):
            self.train = np.array(train["values"])
            self.ds_train = np.array(train["timestamps"])
            self.test = np.array(test["values"])
            self.ds_test = np.array(test["timestamps"])

        def fit_model(self, n_predict):

                fit = SimpleExpSmoothing(self.train).fit()
                forecast = fit.forecast(n_predict)

                ds = self.ds_test
                
                self.forecast = pd.DataFrame({"ds": ds, "yhat": forecast})

                return self.forecast

        def graph(self, metric_name, key):
                plt.figure(figsize=(40,10))
                
                plt.plot(np.array(self.forecast["ds"]), np.array(self.forecast["yhat"]), 'g', label = 'yhat')
                plt.plot(self.ds_train, self.train, 'b', label = 'train', linewidth = 3)
                plt.plot(self.ds_test, self.test, 'k', label = 'test', linewidth = 3)
                # pl.plot(np.array(self.forecast["ds"]), np.array(self.forecast["yhat_upper"]), 'y', label = 'yhat_upper')
                # pl.plot(np.array(self.forecast["ds"]), np.array(self.forecast["yhat_lower"]), 'y', label = 'yhat_lower')
                
                plt.legend()
                plt.savefig("../testing/exp_smoothing_graphs/graph_" + metric_name + "_" + str(key) + ".png")
                plt.show()



def calc_delta(vals):
        diff = vals - np.roll(vals, 1)
        diff[0] = 0
        return diff

def monotonically_inc(vals):
        # check corner case
        if len(vals) == 1:
                return True
        diff = calc_delta(vals)
        diff[np.where(vals == 0)] = 0

        if ((diff < 0).sum() == 0):
                return True
        else:
                return False

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="frun Prophet training on time series")

        parser.add_argument("--metric", type=str, help='metric name', required=True)

        parser.add_argument("--key", type=int, help='key number')
        
        args = parser.parse_args()

        metric_name = args.metric

        pkl_file = open("../pkl_data/" + metric_name + "_dataframes.pkl", "rb")
        dfs = pickle.load(pkl_file)
        pkl_file.close()
        key_vals = list(dfs.keys())

        selected = [args.key]
        for ind in selected:
                key = key_vals[ind]
                df = dfs[key]
                df = df.sort_values(by=['timestamps'])

                print(key)
                df["values"] = df["values"].apply(pd.to_numeric)
                vals = np.array(df["values"].tolist())

                # check if metric is a counter, if so, run AD on difference
                if monotonically_inc(vals):
                        print("monotonically_inc")
                        vals = calc_delta(vals)
                        df["values"] = vals
                
                train = df[0:int(0.7*len(vals))]
                test = df[int(0.7*len(vals)):]

                es = Exp_Smoothing(train, test)
                forecast = es.fit_model(test.shape[0])
                
                f = open("../testing/exp_smoothing_forecasts/forecast_" + metric_name + "_" + str(args.key) + ".pkl", "wb")
                pickle.dump(forecast, f)
                pickle.dump(train, f)
                pickle.dump(test,f)
                f.close()

                es.graph(metric_name, args.key)
