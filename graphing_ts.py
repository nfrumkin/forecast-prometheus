import pickle
import numpy as np
from numpy import fft
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from scipy.stats import chisquare
import collections
import bz2
def fourierExtrapolation(x, n_predict, n_harm):
	n = x.size
	#n_harm = 100                     # number of harmonics in model
	t = np.arange(0, n)
	p = np.polyfit(t, x, 1)         # find linear trend in x
	x_notrend = x - p[0] * t        # detrended x
	x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
	f = fft.fftfreq(n)              # frequencies
	indexes = np.arange(n).tolist()
	# sort indexes by frequency, lower -> higher
	indexes.sort(key = lambda i:np.absolute(f[i]))
 
	t = np.arange(0, n + n_predict)
	restored_sig = np.zeros(t.size)
	for i in indexes[:1 + n_harm * 2]:
		ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		phase = np.angle(x_freqdom[i])          # phase
		restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
	return restored_sig + p[0] * t

def fit_model(train, n_predict):

	model = collections.namedtuple('model',['upper','lower','forecast'])
	
	minimum = np.min(train)
	stddev = np.std(train)

	model.upper = np.max(train) + stddev
	model.lower = minimum - stddev
	if minimum > 0:
		model.lower = max(0, model.lower)

	# n_harm = 1/3 of number of data points was chosen by visual inspection
	n_harm = int(len(train)/3)

	model.forecast = fourierExtrapolation(train, n_predict, n_harm)

	return model

def window_AD(forecast,  test, win_size):
	num_bins = 5

	new_forecast = forecast[-len(test):]
	# windows = [np.arange(win_size*i,win_size*(i+1)) for i in range(int(len(test)/win_size) + 1)]
	# windows[-1] = np.arange(windows[-1][0], len(test))
	win_test = test[1:win_size]
	win_forecast = new_forecast[1:win_size]
	p_vals = []
	for j in range(0, len(test)):
		print(j+len(forecast)-len(test))
		win_test = test[1:win_size]
		for i in range(0,len(test)):
			test_hist, bin_edges = np.histogram(win_test, bins=num_bins)
			big_vals = np.where(win_forecast > bin_edges[-1])
			small_vals = np.where(win_forecast < bin_edges[0])
			f_hist, bin_edges = np.histogram(win_forecast, bins=bin_edges)
			# print(np.sum(test_hist))
			# print(np.sum(f_hist))
			f_hist[-1] = f_hist[-1] + len(big_vals)
			f_hist[0] = f_hist[0] + len(small_vals)
			test_hist = [x+1 for x in test_hist]
			f_hist = [x+1 for x in f_hist]
			# print(test_hist)
			# print(f_hist)
			vals = chisquare(f_hist, f_exp=test_hist )
			# print(vals[1])
			p_vals.append(vals[1])
			# win_test = np.roll(win_test, 1)
			new_forecast = np.roll(new_forecast, 1)
			win_forecast = new_forecast[1:win_size]
		# if p_val > .75:
		# 	return True
		print(np.max(np.array(p_vals)))
		p_vals = []
		test = np.roll(test, 1)
	return False

def detect_anomalies(model, test):
	if np.max(test) > model.upper:
		print("yep")
		#return "point-wise anomaly - upper bound exceeded\nbound: " + str(model.upper) + "\nexceeded value: " + str(np.max(test))
	if np.min(test) < model.lower:
		print('yep')
		#return "point-wise anomaly - lower bound exceeded"
	else:
		# run histogram-based AD
		if window_AD(model.forecast, test, 60):
			return "5-min window anomaly detected"
		return "running histogram-based AD"
	return "no anomalies detected"

def graph(series):
	x_series = np.arange(series.size)
	plt.plot(x_series, series, 'b', label = 'x', linewidth = 3)
	# pl.plot(x_test, test, 'g*', label = 'x', linewidth = 3)
	#pl.plot(x_extrapolation, extrapolation, 'r', label = 'extrapolation')

	plt.legend()
pkl_file = open("../pkl_data/http_request_duration_microseconds_quantile_dataframes.pkl", "rb")
dfs = pickle.load(pkl_file)
pkl_file.close()
print(type(dfs))
key_vals = list(dfs.keys())
print(len(key_vals))


pkl_file = open("../data/real_data_test.pkl", "wb")
pickle.dump(dfs, pkl_file)
pkl_file.close()
i = 0
for key in key_vals[0:800]:
	print(key)
	df = dfs["{'__name__': 'http_request_duration_microseconds', 'beta_kubernetes_io_arch': 'amd64', 'beta_kubernetes_io_instance_type': 'm4.xlarge', 'beta_kubernetes_io_os': 'linux', 'failure_domain_beta_kubernetes_io_region': 'us-east-2', 'failure_domain_beta_kubernetes_io_zone': 'us-east-2a', 'handler': 'prometheus', 'hostname': 'free-stg-node-compute-e0756', 'instance': 'ip-172-31-76-144.us-east-2.compute.internal', 'job': 'kubernetes-nodes-exporter', 'kubernetes_io_hostname': 'ip-172-31-76-144.us-east-2.compute.internal', 'logging_infra_fluentd': 'true', 'node_role_kubernetes_io_compute': 'true', 'quantile': '0.99', 'region': 'us-east-2', 'type': 'compute'}"]
	# df = dfs[key]
	df["values"] = df["values"].apply(pd.to_numeric)
	df = df.sort_values(by=['timestamps'])
	vals = np.array(df["values"].tolist())
	# train = vals[0:int(0.7*len(vals))]
	# test = vals[int(0.7*len(vals)):]
	# print(np.max(test))
	# print(np.where(test == np.max(test)))
	# x_vals = np.arange(0,len(vals))
	# x_test = x_vals[int(0.7*len(vals)):]
	# x_train = x_vals[0:int(0.7*len(vals))]
	# mdl = fit_model(train, len(test))
	print(i)
	i += 1
	# print(detect_anomalies(mdl, test))

	graph(vals)
	# plt.show()
	plt.savefig("../time_series_graphing/graphs_http_total/fourier_" + str(i) + ".png")
	plt.close()