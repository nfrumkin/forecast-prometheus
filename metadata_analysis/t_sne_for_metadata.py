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
from sklearn.manifold import TSNE
import time
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
    X = np.zeros(shape=[1, 200])
    master_labels = []
    label_ints = []
    mds = []
    for one_json in jsons:
        for row in range(0, len(one_json)):
            metadata = one_json[row]["metric"]
            labels = list(metadata.keys())
            label_vals = list(metadata.values())
            x_feature = np.zeros(shape=[1,200])
            for i in range(0, len(labels)):
                flag = True
                for j in range(0,len(master_labels)):
                    if master_labels[j] == labels[i]:
                        if label_vals[i] in label_ints[j]:
                            x_feature[0,j] = label_ints[j][label_vals[i]]
                        else:
                            label_ints[j][label_vals[i]] = len(label_ints[j])+1
                        flag = False
                if flag:
                    master_labels.append(labels[i])
                    label_ints_tmp = {}
                    label_ints_tmp[label_vals[i]] = 1
                    x_feature[0,len(label_ints)] = label_ints_tmp[label_vals[i]]
                    label_ints.append(label_ints_tmp)
                    
            mds.append(metadata)
            X = np.vstack((X, x_feature))
    X = X[1:,:]
    return X, master_labels, label_ints, mds

jsons_bkt = []
for i in range(0,len(filename_bkt)):
        file = filename_bkt[i]
        print(i)
        print(file)
        f = bz2.BZ2File(file, 'rb')
        jsons_bkt.append(json.load(f))
        f.close()
        if i >= 15:
            break

X, master_labels, label_ints, mds = parse_jsons(jsons_bkt, label)

X_embedded = TSNE(n_components=2).fit_transform(X)
file = open("x_vals", "wb")
pickle.dump(X, file)
pickle.dump(X_embedded, file)
pickle.dump(master_labels, file)
pickle.dump(label_ints, file)
pickle.dump(mds, file)
file.close()


print(X_embedded.shape)
plt.figure(figsize=(20,10))
plt.scatter(X_embedded[:,0], X_embedded[:,1],cmap=plt.cm.Spectral)
plt.show()
# fig = plt.figure(figsize=(20,10))
# lbls = set()
# label_axis = {}
# select_label = ""
# while(True):
#     master_md = []
#     times = []
#     times_int = []
#     md = []
#     #filename_bkt = filename_bkt[1:2]
#     if len(filename_bkt) == 0:
#         break
#     print(len(filename_bkt))

#     jsons_bkt = []
#     for i in range(0,len(filename_bkt)):
#         file = filename_bkt[i]
#         print(i)
#         print(file)
#         f = bz2.BZ2File(file, 'rb')
#         jsons_bkt.append(json.load(f))
#         f.close()
#         if i >= 50:
#             break

#     try:
#         filename_bkt = filename_bkt[50:]
#     except:
#         filename_bkt = []

#     master_md, times, md = parse_jsons(jsons_bkt, label)

#     for lbl in master_md:
#         print("\n==", lbl, len(master_md[lbl]))
#         for lbl_val in master_md[lbl]:
#             print("\t", lbl_val)

#     if select_label == "":
#         select_label = input("\n\nSelect a label to graph:\n")
#         try:
#             label_vals = master_md[select_label]
#         except:
#             print("Not a valid label. Exiting..")
#             exit(1)

#     graph = {}
#     for md_i in range(0, len(md)):
#         metadata = md[md_i]
#         try:
#             label_val = metadata[select_label]
#         except:
#             continue

#         try:
#             graph[label_val].extend(times[md_i])
#         except:
#             graph[label_val] = times[md_i]


#     for j in graph.keys():
#         lbls.add(j)
#     print("number of label values: ", len(graph.keys()))

#     for i in lbls:
#         print(i)
#         try:
#             x = dt.date2num(graph[i])
#         except:
#             continue
#         try:
#             val = label_axis[i]
#             y = [val]*len(x)
#         except:
#             val = len(label_axis)
#             label_axis[i] = val
#             y = [val]*len(x)

#         plt.plot(x, y, ',', color=colors[(val+1)%len(colors)])


# del metadata
# del md
# del master_md
# del times

# title = select_label    
# plt.gcf().autofmt_xdate()
# plt.suptitle(m_name)
# plt.title(title)
# plt.xlabel("Timestamp")
# plt.xticks(rotation=25)
# plt.ylabel("Value")
# plt.yticks(np.arange(len(label_axis.keys())))

# ax = plt.gca()
# xfmt = dt.DateFormatter('%Y-%m-%d %H:%M:%S')
# ax.xaxis.set_major_formatter(xfmt)
# #plt.show()
# plt.savefig(png_name)
# plt.close()

# # plot the legend table
# plt.figure(figsize=(20,10))
# print(label_axis.keys())
# n_lbls = np.array(list(label_axis.keys()))
# n_lbls.shape = (len(lbls), 1)
# vals = np.array(list(label_axis.values()))
# vals.shape = (len(vals), 1)
# table_vals = np.append(vals, n_lbls, 1)
# t = plt.table(cellText=table_vals, colLabels=["Number", label], cellLoc='center', loc='center')
# # t.set_fontsize(18)
# t.scale(1,3)
# plt.axis("off")
# plt.title("LEGEND")
# plt.savefig(png_name2)