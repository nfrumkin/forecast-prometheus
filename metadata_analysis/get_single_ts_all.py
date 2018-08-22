metadata = {'__name__': 'kubelet_docker_operations_latency_microseconds', 'beta_kubernetes_io_arch': 'amd64', 'beta_kubernetes_io_instance_type': 'm4.xlarge', 'beta_kubernetes_io_os': 'linux', 'clam_controller_enabled': 'True', 'clam_server_enabled': 'True', 'failure_domain_beta_kubernetes_io_region': 'us-east-2', 'failure_domain_beta_kubernetes_io_zone': 'us-east-2a', 'fluentd_test': 'true', 'hostname': 'free-stg-master-5c6a9', 'instance': 'ip-172-31-73-251.us-east-2.compute.internal', 'job': 'kubernetes-nodes', 'kubernetes_io_hostname': 'ip-172-31-73-251.us-east-2.compute.internal', 'node_role_kubernetes_io_compute': 'true', 'node_role_kubernetes_io_master': 'true', 'operation_type': 'version', 'quantile': '0.5', 'region': 'us-east-2', 'type': 'master'}

import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import pandas as pd
import re
import string 
import random
import numpy as np
import fnmatch
import os
import sys
import bz2
import pickle
if len(sys.argv) != 2:
    print("incorrect number of command line arguments")
    print("received: ", len(sys.argv))
    print("expected: 2")
    exit(1)

m_name = sys.argv[1]
metric_type = ""
data_folder = "../data/"

# find bucket/quantile, sum, and count files in metric folder
filename_bkt = []

try:
    for file in os.listdir(data_folder + m_name + "/bucket/"):
        if fnmatch.fnmatch(file, "*.json.bz2"):
            metric_type = "hist"
            f_name = data_folder + m_name + "/bucket/" + file
            filename_bkt.append(f_name)
except:
    for file in os.listdir(data_folder + m_name + "/quantile/quant2/"):
        if fnmatch.fnmatch(file, "*.json"):
            metric_type = "summary"
            f_name = data_folder + m_name + "/quantile/quant2/" + file
            filename_bkt.append(f_name)

print("Metric: ", m_name)
if metric_type == "hist":
    label = "le"
elif metric_type == "summary":
    label = "quantile"
else:
    print("no metric type detected")
    exit(1)

print("Metric: ", m_name)
if metric_type == "hist":
    label = "le"
elif metric_type == "summary":
    label = "quantile"
else:
    print("no metric type detected")
    exit(1)

results_folder = "../results/"


# load appropriate data
inc = 0
num_jsons = 10
jsons_bkt = []
print(len(filename_bkt))
dfs = []
for file in filename_bkt:
    f = open(file, 'rb')
    # f = bz2.BZ2File(file, 'rb')
    one_json = json.load(f)
    f.close()
    for pkt in one_json:
            df = pd.DataFrame.from_dict(pkt["values"])
            df = df.rename( columns={0:"timestamps", 1:"values"})
            df["timestamps"] = pd.to_datetime(df["timestamps"], unit='s')
            df = df.sort_values(by=["timestamps"])
            meta_keys = np.array(list(pkt["metric"].keys()))
            meta_vals = np.array(list(pkt["metric"].values()))
            md = dict(zip(meta_keys, meta_vals))
            if md == metadata:
            	dfs.append(df)
            	print(len(dfs))
    # if inc == num_jsons:
    #     break
    print(inc)
    inc += 1


file = "df_one_ts" + ".pkl"
pickle_file = open(file, "wb")
pickle.dump(dfs, pickle_file)
pickle_file.close()


