import pickle
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime as dt
import argparse

class ProphetForecast:
	def __init__(self, train, test):
		self.train = train
		self.test = test

	def fit_model(self, n_predict):
		m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
		m.fit(self.train)
		future = m.make_future_dataframe(periods= len(self.test),freq= '1MIN')
		self.forecast = m.predict(future)

		return self.forecast

	def graph(self):
		fig = plt.figure(figsize=(40,10))
		plt.plot(np.array(self.train["ds"]), np.array(self.train["y"]),'b', label="train", linewidth=3)
		plt.plot(np.array(self.test["ds"]), np.array(self.test["y"]), 'g', label="test", linewidth=3)

		forecast_ds = np.array(self.forecast["ds"])
		plt.plot(forecast_ds, np.array(self.forecast["yhat"]), 'o', label="yhat", linewidth=3)
		plt.plot(forecast_ds, np.array(self.forecast["yhat_upper"]), 'y', label="yhat_upper", linewidth=3)
		plt.plot(forecast_ds, np.array(self.forecast["yhat_lower"]), 'y', label="yhat_lower", linewidth=3)
		plt.xlabel("Timestamp")
		plt.ylabel("Value")
		plt.legend(loc=1)
		plt.title("Prophet Model Forecast")
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

	parser = argparse.ArgumentParser(description="run Prophet training on time series")

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

		df["ds"] = df["timestamps"]
		df["y"] = df["values"]
		# check if metric is a counter, if so, run AD on difference
		if monotonically_inc(vals):
			print("monotonically_inc")
			vals = calc_delta(vals)
			df["values"] = vals.tolist()
		
		train = df[0:int(0.7*len(vals))]
		test = df[int(0.7*len(vals)):]

		pf = ProphetForecast(train, test)
		forecast = pf.fit_model(len(test))

		f = open("../prophet_forecasts/prophet_model_" + metric_name + "_" + str(args.key) + ".pkl", "wb")
		pickle.dump(forecast,f)
		pickle.dump(train, f)
		pickle.dump(test,f)
		f.close()
		
		pf.graph()


