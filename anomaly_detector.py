import pickle
import pylab as pl
import numpy as np
from scipy.stats import norm

import matplotlib.transforms as mtransforms

class Accumulator:
	def __init__(self,thresh):
		self._counter = 0
		self.thresh = thresh
	def inc(self, val):
		self._counter += val
	def count(self):
		return self._counter

class AnomalyDetector:
	def __init__(self, window=8000, small_window=80, epsilon=0.61, bounds_thresh=22000, peak_thresh=130000, acc_thresh=1000):
		# accumulator parameters
		self.large_window = window
		self.small_window = small_window
		self.epsilon = epsilon
		# tail probability parameters
		self.bounds_thresh = bounds_thresh
		self.peak_thresh = peak_thresh
		self.acc_thresh = acc_thresh

	def anomaly_tail_distribution(self, w, w_prime):
		if len(w) != self.large_window:
			return "ERROR: input values do not match window size"
		mu = np.mean(w)
		std = np.std(w)
		mu_bar = np.mean(w_prime)
		
		L_t = norm.pdf(((mu_bar - mu)/std))
		# print(L_t)
		if L_t >= 1 - self.epsilon:
			return 1
		return 0

	def anomaly_accumulator(self, y, y_hat):
		s_t = []
		anomaly_inds = []
		acc_thresh = self.acc_thresh
		acc = Accumulator(acc_thresh)
		for i in range(0, len(y_hat)):
			diff = y_hat[i] - y[i]
			if abs(diff) > self.bounds_thresh:
				# upper bound anomaly, increment counter
				acc.inc(1)
			elif y[i] > self.peak_thresh:
				# found peak, decrement so that acc will decay to 0
				acc.inc(-3)
			else:
				# no anomaly, decrement by 2
				acc.inc(-2)

			if acc.count() > acc.thresh:
				anomaly_inds.append(i)

			s_t.append(max(diff, 0))
		return s_t, anomaly_inds
	def get_anomalies(self, y, y_hat):
		if len(y) != len(y_hat):
			return "ERROR: lengths of inputs do not match"
		s_t, anomaly_inds_acc = self.anomaly_accumulator(y, y_hat)
		cum_window = self.large_window+self.small_window
		
		anomaly_inds_tail = []
		print("st:", len(s_t))
		print("cum_wind:", cum_window)
		for i in range(0,(len(s_t)-cum_window)):
			window = s_t[i:int(i+self.large_window)]
			small_window = s_t[int(i+self.large_window):int(i+cum_window)]
			val = self.anomaly_tail_distribution(window, small_window)
			anomaly_inds_tail.append(val)
		anomaly_inds_tail = np.argwhere(anomaly_inds_tail).flatten()

		print("a_i_tail: ", len(anomaly_inds_tail))
		print("a_i_accum: ", len(anomaly_inds_acc))
		# get intersection of both 
		set_tail = set(anomaly_inds_tail)
		set_acc = set(anomaly_inds_acc)
		flag_anomaly = set_tail.intersection(set_acc)
		return flag_anomaly

def graph(train, test, extrapolation, anomalies, metric_name):
	len_train = len(train)
	fig = pl.figure(figsize=(20,10))
	x_extrapolation = np.arange(0, extrapolation.size)
	x_train = np.arange(0, train.size)
	x_test = np.arange(train.size , extrapolation.size)
	ax = pl.axes() 
	ax.plot(x_train, train, 'b', label = 'train', linewidth = 3)
	ax.plot(x_test, test, 'g', label = 'test', linewidth = 3)
	ax.plot(x_test, extrapolation[x_train.size:x_extrapolation.size], 'y', label = 'forecast')
	title = "Forecast for " + metric_name
	ax.set_title(title)
	ax.set_xlabel("Timestamp")
	ax.set_ylabel("Value")
	trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
	for a in anomalies:
		bool_arr = x_test>(a-100+len_train)
		bool_arr2 = x_test<(a+len_train)
		bool_arr3 = np.logical_and(bool_arr, bool_arr2)
		ax.fill_between(x_test,0,1, where=bool_arr3, facecolor='red', alpha=0.5, transform=trans)
	pl.legend(loc=3)
	pl.show()

metric_name = "http_request_duration_microseconds_quantile"
filename = "../forecasts/forecast_" + metric_name + ".pkl"
pkl_file = open(filename, "rb")
forecast = pickle.load(pkl_file)
train = pickle.load(pkl_file)
test = pickle.load(pkl_file)
pkl_file.close()

ad = AnomalyDetector()
anomaly_inds = ad.get_anomalies(test, forecast[-len(test):])
graph(train, test, forecast, anomaly_inds, metric_name)