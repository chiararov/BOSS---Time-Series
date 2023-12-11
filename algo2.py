import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from algo1 import BOSSTransform

w = 5
l = 4
c = 2
mean = False

def TS_predict(test, samples, w,l,c):
    bestDist=np.inf
    bestTs=None
    n=len(samples)
    h= BOSSTransform(test, w, l, c, mean=False)
    for i in range(n):
        signal=samples.iloc[i]
        hi=BOSSTransform(signal, w, l, c, mean=False)
        dist=boss_distance(h,hi)
        if dist < bestDist:
            bestDist = dist
            bestTs = signal
    return bestTs

def boss_distance(h1,h2):
    dist=0
    for word in h1.keys():
        if word in h2.keys():
            dist+=(h1[word]-h2[word])**2
        else:
            dist+=h1[word]**2
    return dist

path = 'StarLightCurves/StarLightCurves_TRAIN.txt'
data_with_label = pd.read_fwf(path, header=None)
data_without_label = data_with_label.iloc[:, 1:]
sample1 = data_without_label.iloc[0]
sample2 = data_without_label.iloc[1]

print(TS_predict(sample1,data_without_label,w,l,c))
