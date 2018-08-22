import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import re
import string 
import random
import numpy as np
import bz2
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('label_hists2.pdf')
import os

label = "instance"
folder = "kubelet_docker_operations_latency_microseconds/"
files = os.listdir(folder)
jsons = []

inc = 0
print(len(files))
md = []
for file in files:
    inc += 1
    print(inc)
    filen = folder + file
    try:
        f = bz2.BZ2File(filen, 'rb')
        jsonFile = json.load(f)
        f.close()
    except IsADirectoryError:
        continue
    for pkt in jsonFile:
        metadata = pkt["metric"]
        del metadata["__name__"]
        md.append(metadata)

lbls = {}
for i in range(0, len(md)):
    for key in md[i].keys():
        if key in lbls.keys():
            lbls[key].append(md[i][key])
        else:
            lbls[key] = [md[i][key]]

for key in lbls.keys():
    vals = lbls[key]
    plt.figure(figsize=(10,5))
    plt.hist(vals)
    #plt.gcf().autofmt_xdate()
    #plt.legend(lbl)
    plt.title(key)
    plt.xlabel("Label Value")
    plt.ylabel("Count")
    plt.savefig(pp, format='pdf')
    plt.close()

pp.close()