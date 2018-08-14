import pickle
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime as dt

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

def fit_model(train, test, n_predict):
	m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
	#m = Prophet()
	m.fit(train)
	future = m.make_future_dataframe(periods= len(test),freq= '1MIN')
	forecast = m.predict(future)
	forecast.head()
	# forecasted_features = ['ds','yhat','yhat_lower','yhat_upper']
	# fig = plt.figure()
	# ax = plt.scatter(df['ds'], df['y'], color='c')
	#m.plot(forecast,xlabel="Timestamp",ylabel="Value")
	return m, forecast

def graph(train_df, test_df, forecast):
	fig = plt.figure(figsize=(40,10))
	plt.plot(np.array(train_df["ds"]), np.array(train_df["y"]),'b', label="train", linewidth=3)
	plt.plot(np.array(test_df["ds"]), np.array(test_df["y"]), 'g', label="test", linewidth=3)
	print("TRAIN")
	print(train_df["ds"])
	print("\n\nTEST")
	print(test_df["ds"])
	forecast_ds = np.array(forecast["ds"])
	print("\n\nFORECAST")
	print(forecast_ds)
	plt.plot(forecast_ds, np.array(forecast["yhat"]), 'o', label="yhat", linewidth=3)
	plt.plot(forecast_ds, np.array(forecast["yhat_upper"]), 'y', label="yhat_upper", linewidth=3)
	plt.plot(forecast_ds, np.array(forecast["yhat_lower"]), 'y', label="yhat_lower", linewidth=3)
	plt.xlabel("Timestamp")
	plt.ylabel("Value")
	plt.legend(loc=1)
	plt.title("Prophet Model Forecast")
	plt.show()


metric_name = "http_request_duration_microseconds_quantile"
pkl_file = open("../pkl_data/" + metric_name + "_dataframes.pkl", "rb")
dfs = pickle.load(pkl_file)
pkl_file.close()
key_vals = list(dfs.keys())

for key in key_vals[728:729]:
	df = dfs[key]
	print(key)
	df["values"] = df["values"].apply(pd.to_numeric)
	vals = np.array(df["values"].tolist())

	# check if metric is a counter, if so, run AD on difference
	if monotonically_inc(vals):
		print("monotonically_inc")
		vals = calc_delta(vals)
	train = vals[0:int(0.7*len(vals))]
	test = vals[int(0.7*len(vals)):]
	print(np.max(test))
	print(np.where(test == np.max(test)))
	x_vals = np.arange(0,len(vals))
	x_test = x_vals[int(0.7*len(vals)):]
	x_train = x_vals[0:int(0.7*len(vals))]

	train_dict = {}
	train_dict["y"] = train
	train_dict["ds"] = pd.date_range(start=dt.datetime(2018,5,10,7,8), periods=len(train), freq="1MIN").tolist()
	train_df = pd.DataFrame(train_dict)
	print(train_df.head())
	mdl, forecast = fit_model(train_df, test, len(test))

	test_dict = {}
	test_dict["y"] = test
	test_dict["ds"] = pd.date_range(start=max(train_dict["ds"]), periods=len(test), freq="1MIN").tolist()
	test_df = pd.DataFrame(test_dict)

	f = open("prophet_model_" + metric_name + ".pkl", "wb")
	pickle.dump(mdl, f)
	pickle.dump(forecast,f)
	pickle.dump(train_df, f)
	pickle.dump(test_df,f)
	f.close()
	
	graph(train_df, test_df, forecast)


