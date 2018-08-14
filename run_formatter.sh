#!/usr/bin/env bash

time python format_to_pandas.py \
       	--metric http_requests_total \
	--input ../data/ \
	--output ../results/ \
	--batch_size 20
