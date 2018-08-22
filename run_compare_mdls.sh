#! /usr/bin/env bash

METRIC='http_request_duration_microseconds_quantile'
KEY=60
python prophet_train.py --metric $METRIC --key $KEY
python fourier_train.py --metric $METRIC --key $KEY
python compare_fourier_prophet.py --metric $METRIC --key $KEY
