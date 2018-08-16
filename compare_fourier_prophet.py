import pickle
from matplotlib.pylab import plt
import numpy as np
import argparse

def graph(train_df, test_df, p_forecast, f_forecast):
	fig = plt.figure(figsize=(40,10))
	forecast_ds = np.array(p_forecast["ds"])
	plt.plot(forecast_ds, f_forecast, 'C1', label="fourier_yhat", linewidth=3)

	plt.plot(np.array(train_df["ds"]), np.array(train_df["y"]),'*k', label="train", linewidth=3)
	plt.plot(np.array(test_df["ds"]), np.array(test_df["y"]), '*b', label="test", linewidth=3)

	plt.plot(forecast_ds, np.array(p_forecast["yhat"]), '*y', label="prophet_yhat", linewidth=3)
	plt.plot(forecast_ds, np.array(p_forecast["yhat_upper"]), 'y', label="yhat_upper", linewidth=3)
	plt.plot(forecast_ds, np.array(p_forecast["yhat_lower"]), 'y', label="yhat_lower", linewidth=3)
	
	
	plt.plot()
	plt.xlabel("Timestamp")
	plt.ylabel("Value")
	plt.legend(loc=1)
	plt.title("Prophet Model Forecast")
	plt.savefig( "../testing/compare_fourier_prophet/compare_" + args.metric + "_" + str(args.key) + ".png")
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="run Fourier training on time series")

	parser.add_argument("--metric", type=str, help='metric name', required=True)
	parser.add_argument("--key", type=int, help='key number')

	args = parser.parse_args()
	
	fname = "../prophet_forecasts/prophet_model_" + args.metric + "_" + str(args.key) + ".pkl"
	f = open(fname, "rb")
	p_forecast = pickle.load(f)
	p_train = pickle.load(f)
	p_test = pickle.load(f)
	f.close()

	fname = "../fourier_forecasts/forecast_" + args.metric + "_" + str(args.key) + ".pkl"
	f = open(fname, "rb")
	f_forecast = pickle.load(f)
	f_train = pickle.load(f)
	f_test = pickle.load(f)
	f.close()

	graph(p_train, p_test, p_forecast, f_forecast)
