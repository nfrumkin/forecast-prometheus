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
import bz2
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import EventCollection
import time

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
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
    for file in os.listdir(data_folder + m_name + "/quantile/"):
        if fnmatch.fnmatch(file, "*.json.bz2"):
            metric_type = "summary"
            f_name = data_folder + m_name + "/quantile/" + file
            filename_bkt.append(f_name)

print("Metric: ", m_name)
if metric_type == "hist":
    label = "le"
elif metric_type == "summary":
    label = "quantile"
else:
    print("no metric type detected")
    exit(1)

results_folder = "../results/"
png_name = results_folder + m_name + '_graphs.png'
png_name2 = results_folder + m_name + '_graphs_legend.png'

def parse_jsons(jsons, select_label="__name__"):
    times = []
    master_md = {}
    md = []
    for one_json in jsons:
        for pkt in one_json:
            timestamps = []
            timestamps_int = []
            for i in pkt["values"]:
                timestamps.append(datetime.fromtimestamp(float(i[0])))
            metadata = pkt["metric"]
            for lbl in metadata.keys():
                lbl_val = metadata[lbl]
                if lbl in master_md.keys():
                    if lbl_val in master_md[lbl]:
                        continue
                    else:
                        master_md[lbl].append(lbl_val)
                else:
                    temp_list = []
                    temp_list.append(lbl_val)
                    master_md[lbl] = temp_list
            times.append(timestamps)
            md.append(metadata)
    return master_md, times, md


fig = plt.figure(figsize=(20,10))
lbls = set()
label_axis = {}
select_label = ""
while(True):
    master_md = []
    times = []
    times_int = []
    md = []
    #filename_bkt = filename_bkt[1:2]
    if len(filename_bkt) == 0:
        break
    print(len(filename_bkt))

    jsons_bkt = []
    for i in range(0,len(filename_bkt)):
        file = filename_bkt[i]
        print(i)
        print(file)
        f = bz2.BZ2File(file, 'rb')
        jsons_bkt.append(json.load(f))
        f.close()
        if i >= 50:
            break

    try:
        filename_bkt = filename_bkt[50:]
    except:
        filename_bkt = []

    master_md, times, md = parse_jsons(jsons_bkt, label)

    for lbl in master_md:
        print("\n==", lbl, len(master_md[lbl]))
        for lbl_val in master_md[lbl]:
            print("\t", lbl_val)

    if select_label == "":
        select_label = input("\n\nSelect a label to graph:\n")
        try:
            label_vals = master_md[select_label]
        except:
            print("Not a valid label. Exiting..")
            exit(1)

    graph = {}
    for md_i in range(0, len(md)):
        metadata = md[md_i]
        try:
            label_val = metadata[select_label]
        except:
            continue

        try:
            graph[label_val].extend(times[md_i])
        except:
            graph[label_val] = times[md_i]


    for j in graph.keys():
        lbls.add(j)
    print("number of label values: ", len(graph.keys()))

    for i in lbls:
        print(i)
        try:
            x = dt.date2num(graph[i])
        except:
            continue
        try:
            val = label_axis[i]
            y = [val]*len(x)
        except:
            val = len(label_axis)
            label_axis[i] = val
            y = [val]*len(x)

        plt.plot(x, y, ',', color=colors[(val+1)%len(colors)])


del metadata
del md
del master_md
del times

title = select_label    
plt.gcf().autofmt_xdate()
plt.suptitle(m_name)
plt.title(title)
plt.xlabel("Timestamp")
plt.xticks(rotation=25)
plt.ylabel("Value")
plt.yticks(np.arange(len(label_axis.keys())))

ax = plt.gca()
xfmt = dt.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
#plt.show()
plt.savefig(png_name)
plt.close()

# plot the legend table
plt.figure(figsize=(20,10))
print(label_axis.keys())
n_lbls = np.array(list(label_axis.keys()))
n_lbls.shape = (len(lbls), 1)
vals = np.array(list(label_axis.values()))
vals.shape = (len(vals), 1)
table_vals = np.append(vals, n_lbls, 1)
t = plt.table(cellText=table_vals, colLabels=["Number", label], cellLoc='center', loc='center')
# t.set_fontsize(18)
t.scale(1,3)
plt.axis("off")
plt.title("LEGEND")
plt.savefig(png_name2)