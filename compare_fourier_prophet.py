import pickle
from matplotlib.pylab import plt
import numpy as np
import argparse

def graph(train_df, test_df, p_forecast, f_forecast, metric, key):
	fig = plt.figure(figsize=(40,10))
	forecast_ds = np.array(f_forecast["ds"])
	print(len(forecast_ds))
	print(len(train_df))
	forecast_ds = forecast_ds[int(train_df["values"].count()):]


	plt.plot(np.array(train_df["ds"]), np.array(train_df["y"]),'b', label="train", linewidth=3)
	plt.plot(np.array(test_df["ds"]), np.array(test_df["y"]), 'k', label="test", linewidth=3)

	plt.savefig( "../testing/compare_fourier_prophet/" + str(key) + "_raw_" + metric + ".png", transparent=True)
	prophet = np.array(p_forecast["yhat"])
	prophet_upper = np.array(p_forecast["yhat_upper"])
	prophet_lower = np.array(p_forecast["yhat_lower"])

	fourier = f_forecast["yhat"]
	fourier = fourier[len(train_df["values"]):]
	print(len(forecast_ds))
	print(len(fourier))
	plt.plot(forecast_ds, fourier, 'g', label="fourier_yhat", linewidth=3)
	plt.savefig( "../testing/compare_fourier_prophet/" + str(key) + "_fourier_" + metric + ".png", transparent=True)

	prophet = prophet[len(train_df["values"]):]
	prophet_upper = prophet_upper[len(train_df["values"]):]
	prophet_lower = prophet_lower[len(train_df["values"]):]
	plt.plot(forecast_ds, prophet, '*y', label="prophet_yhat", linewidth=3)
	plt.plot(forecast_ds, prophet_upper, 'y', label="yhat_upper", linewidth=3)
	plt.plot(forecast_ds, prophet_lower, 'y', label="yhat_lower", linewidth=3)
	
	
	plt.plot()
	plt.xlabel("Timestamp")
	plt.ylabel("Value")
	plt.legend(loc=1)
	plt.title("Prophet Model Forecast")
	plt.savefig( "../testing/compare_fourier_prophet/" + str(key) + "_compare_" + metric + ".png", transparent=True)
	plt.close()


	fig = plt.figure(figsize=(40,10))
	forecast_ds = np.array(f_forecast["ds"])
	forecast_ds = forecast_ds[len(train_df["values"]):]


	plt.plot(np.array(train_df["ds"]), np.array(train_df["y"]),'b', label="train", linewidth=3)
	plt.plot(np.array(test_df["ds"]), np.array(test_df["y"]), 'k', label="test", linewidth=3)

	prophet = np.array(p_forecast["yhat"])
	prophet_upper = np.array(p_forecast["yhat_upper"])
	prophet_lower = np.array(p_forecast["yhat_lower"])
	prophet = prophet[len(train_df["values"]):]
	prophet_upper = prophet_upper[len(train_df["values"]):]
	prophet_lower = prophet_lower[len(train_df["values"]):]
	plt.plot(forecast_ds, prophet, '*y', label="prophet_yhat", linewidth=3)
	plt.plot(forecast_ds, prophet_upper, 'y', label="yhat_upper", linewidth=3)
	plt.plot(forecast_ds, prophet_lower, 'y', label="yhat_lower", linewidth=3)
	plt.savefig( "../testing/compare_fourier_prophet/" + str(key) + "_prophet_" + metric + ".png", transparent=True)
	plt.close()
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="run Fourier training on time series")

	parser.add_argument("--metric", type=str, help='metric name', required=True)
	parser.add_argument("--key", type=int, help='key number')

	args = parser.parse_args()
	
	fname = "../prophet_forecasts/prophet_model_" + args.metric + "_" + str(args.key) + ".pkl"
	f = open(fname, "rb")
	p_forecast = pickle.load(f)
	print(len(p_forecast))
	p_train = pickle.load(f)
	print(len(p_train))
	p_test = pickle.load(f)
	print(len(p_test))
	f.close()

	fname = "../fourier_forecasts/forecast_" + args.metric + "_" + str(args.key) + ".pkl"
	f = open(fname, "rb")
	f_forecast = pickle.load(f)
	print(len(f_forecast))
	f_train = pickle.load(f)
	print(len(f_train))
	f_test = pickle.load(f)
	print(len(f_test))
	f.close()

	graph(p_train, p_test, p_forecast, f_forecast, args.metric, args.key)
