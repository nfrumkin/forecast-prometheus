import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import re
import string 
import random
import numpy as np
import fnmatch
import os
import sys
import ast
from matplotlib.backends.backend_pdf import PdfPages

if len(sys.argv) != 2:
    print("incorrect number of command line arguments")
    print("received: ", len(sys.argv))
    print("expected: 2")
    exit(1)

file = sys.argv[1]
lines = [line.rstrip('\n') for line in open(file)]
m_name = lines[0]
target_metadata = lines[1]
target_metadata = target_metadata.replace("'", "\"")
target_metadata = json.loads(target_metadata)

data_folder = "../data/"
metric_type = "hist"
#metadata = "{'beta_kubernetes_io_arch': 'amd64', 'beta_kubernetes_io_fluentd_ds_ready': 'true', 'beta_kubernetes_io_instance_type': 'm4.xlarge', 'beta_kubernetes_io_os': 'linux', 'clam_controller_enabled': 'True', 'failure_domain_beta_kubernetes_io_region': 'us-east-2', 'failure_domain_beta_kubernetes_io_zone': 'us-east-2a', 'fluentd_test': 'true', 'hostname': 'free-stg-master-03fb6', 'instance': 'ip-172-31-78-254.us-east-2.compute.internal', 'job': 'kubernetes-nodes', 'kubernetes_io_hostname': 'ip-172-31-78-254.us-east-2.compute.internal', 'node_role_kubernetes_io_master': 'true', 'operation_type': 'list_images', 'region': 'us-east-2', 'type': 'master'}"

# find bucket/quantile, sum, and count files in metric folder
for file in os.listdir(data_folder + m_name + "/"):
    if fnmatch.fnmatch(file, "bucket_*.json"):
        metric_type = "hist"
        filename_bkt = data_folder + m_name + "/" + file
    elif fnmatch.fnmatch(file, "quantile_*.json"):
        metric_type = "summary"
        filename_bkt = data_folder + m_name + "/" + file
    if fnmatch.fnmatch(file, "count_*.json"):
        filename_cnt = data_folder + m_name + "/" + file
    if fnmatch.fnmatch(file, "sum_*.json"):
        filename_sum = data_folder + m_name + "/" + file
if metric_type == "hist" or metric_type == "summary":
    print("Metric: ", m_name)
else:
    print("no metric type detected")
    exit(1)

results_folder = "../results/"
pp_graph = PdfPages(results_folder + m_name + '_graphs.pdf')
pp_hist = PdfPages(results_folder + m_name + '_hists.pdf')

# load appropriate data
f = open(filename_bkt)
jsonFile_bkt = json.load(f)
f.close()

f2 = open(filename_cnt)
jsonFile_cnt = json.load(f2)
f2.close()

f3 = open(filename_sum)
jsonFile_sum = json.load(f3)
f3.close()

# each index corresponds to one graph
# each graph is a list of lists
# list of list of lists
# graph = all_series[i]
# one_series = graph[i]
# one_data_point = one_series[i]
b_val = []
b_time = []
b_md = []
b_le = []
for pkt in jsonFile_bkt:
    timestamps = []
    vals = []
    for i in pkt["values"]:
        if i[1] != 'NaN':
            vals.append(float(i[1]))
            timestamps.append(datetime.fromtimestamp(float(i[0])))
    metadata = pkt["metric"]
    if metric_type == "hist":
        le = metadata["le"]
        del metadata["le"]
    elif metric_type == "summary":
        le = metadata["quantile"]
        del metadata["quantile"]
    metric_name = metadata["__name__"]
    del metadata["__name__"]
    if metadata == target_metadata:
	    metadata = str(metadata)
	    if len(vals) > 0:
	        b_val.append(vals)
	        b_time.append(timestamps)
	        b_md.append(metadata)
	        b_le.append(le)

s_val = []
s_time = []
s_md = []
for pkt in jsonFile_sum:
    timestamps = []
    vals = []
    for i in pkt["values"]:
        if i[1] != 'NaN':
            vals.append(float(i[1]))
            timestamps.append(datetime.fromtimestamp(float(i[0])))
    metadata = pkt["metric"]
    metric_name = metadata["__name__"]
    del metadata["__name__"]
    metadata = str(metadata)
    if len(vals) > 0:
        s_val.append(vals)
        s_time.append(timestamps)
        s_md.append(metadata)

c_val = []
c_time = []
c_md = []
for pkt in jsonFile_cnt:
    timestamps = []
    vals = []
    for i in pkt["values"]:
        if i[1] != 'NaN':
            vals.append(float(i[1]))
            timestamps.append(datetime.fromtimestamp(float(i[0])))
    metadata = pkt["metric"]
    metric_name = metadata["__name__"]
    del metadata["__name__"]
    metadata = str(metadata)
    if len(vals) > 0:
        c_val.append(vals)
        c_time.append(timestamps)
        c_md.append(metadata)



graphs = {}
graph_label = []
graph_xs = {}
for md_i in range(0,len(b_md)):
    metadata = str(b_md[md_i])
    label = b_le[md_i]
    try:
        graphs[metadata][label].extend(b_val[md_i])
        graph_xs[metadata][label].extend(b_time[md_i])
    except:
        try:
            graphs[metadata][label] = b_val[md_i]
            graph_xs[metadata][label] = b_time[md_i]
        except:
            label_dict = {}
            label_dict[label] = b_val[md_i]
            label_t_dict = {}
            label_t_dict[label] = b_time[md_i]
            graphs[metadata] = label_dict
            graph_xs[metadata] = label_t_dict



inc = 0
print("number of graphs: ", len(graphs.keys()))
for i in graphs.keys():
    if (inc+1) % 50 == 0:
        pp_graph.close()
        pp_graph = PdfPages(results_folder + str(inc+1) + "_" + m_name + '_graphs.pdf')
        pp_hist.close()
        pp_hist = PdfPages(results_folder + str(inc+1) + "_" + m_name + '_hists.pdf')
    print(inc)
    graph_title = i
    xs = graph_xs[i]
    ys = graphs[i]
    #if graph_title == "{'instance': '172.31.65.74:8444', 'job': 'kubernetes-controllers', 'request': 'detach_volume'}":

    title = re.sub("(.{200})", "\\1\n", graph_title, 0, re.DOTALL)
    if len(graph_title) > 50:
        graph_title= graph_title[1:50]
    plt.figure(figsize=(20,10))
    for j in ys.keys():
        plt.plot(xs[j], ys[j], '*')
    plt.gcf().autofmt_xdate()
    plt.suptitle(metric_name)
    plt.title(title)
    plt.legend(ys.keys())
    plt.xlabel("Timestamp")
    plt.ylabel("Value")

    #savefile = "graphs/" + insts[i] + "_" + graph_title + ".png"
    plt.savefig(pp_graph, format='pdf')
    plt.close()

    main_title = re.sub("(.{200})", "\\1\n", graph_title, 0, re.DOTALL)
    if len(graph_title) > 50:
        graph_title= graph_title[1:50]
    plt.figure(figsize=(20,10))
    for j in ys.keys():
        time = xs[j][0]
        break
    for j in range(0, len(s_time[inc])):
        if s_time[inc][j] == time:
            sum_val = s_val[inc][j]
            break
    for j in range(0, len(c_time[inc])):
        if c_time[inc][j] == time:
            count_val = c_val[inc][j]
            break


    graph_label = list(xs.keys())
    tmp = graph_label
    tmp.sort()
    if metric_type == "hist":
        inf = tmp[0]

        # take away the +Inf bucket
        tmp = tmp[1::]

        # sort the remaining integers/floats
        tmp.sort(key=float)

        # append +Inf to the end
        tmp.append(inf)

    sorted_y = []
    for j in tmp:
        for k in graph_label:
            if j == k:
                sorted_y.append(ys[k][0])
                break

    graph_label = tmp
    bar_vals = np.arange(len(graph_label))
    plt.bar(bar_vals, height =sorted_y)
    plt.xticks(bar_vals, graph_label)
    plt.gcf().autofmt_xdate()
    plt.suptitle(main_title)
    title = "Count: " + str(count_val) + ", Sum: " + str(sum_val)
    plt.title(title, fontsize=20)
    plt.xlabel("Bucket")
    plt.ylabel("Value" )

    # #savefile = "hists/" + insts[i] + ".png"
    plt.savefig(pp_hist, format='pdf')
    plt.close()
    inc += 1

pp_graph.close()
pp_hist.close()