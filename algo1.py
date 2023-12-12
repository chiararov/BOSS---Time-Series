import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rd
# Import Data
path = 'StarLightCurves/StarLightCurves_TRAIN.txt'
data_with_label = pd.read_fwf(path, header=None)
data_without_label = data_with_label.iloc[:, 1:]
sample = data_without_label.iloc[0]  # first sample of length 1024
sample = sample.tolist()

# Adjust parameters
w = 5
l = 4
c = 3
mean = False

# Algo 1


def BOSSTransform(sample, w, l, c, mean):
    words = window_to_word(sample, w, l, c, mean)
    unique_words = np.unique(words)
    boss = {value: 0 for value in unique_words}
    lastword = None
    for word in words:
        if word != lastword:
            boss[word] += 1
        lastword = word
    return boss


def sliding_windows(sample, w):
    'returns list of the n-w sliding windows of length w of a signal'
    n = len(sample)
    s = []
    for i in range(n-w):
        s.append(sample[i:i+w])
    std=np.std(s) #normalisation
    return s/std


def DFT(sample, l, mean):
    n = len(sample)
    dft = []
    if mean:
        sample=sample-np.mean(sample)
    num = int(round(l/2))
    for u in range(num):
        xu = (1/n) * sum(sample[x]*np.exp(- 2j * np.pi*u*x/n)
                         for x in range(n))
        dft.append(xu.real)
        dft.append(xu.imag)
    return dft

def DFT_A(sample,w, l, mean):
    'returns the matrix A for which each lign i is the DFT of the window i of a signal'
    if mean:
        sample=sample-np.mean(sample)

    num = int(round(l/2))
    windows=sliding_windows(sample,w)
    n = len(sample)

    v=np.exp( 2j * np.pi * np.arange(num) /n)
    V=np.diag(v)
    
    dft0=[]
    for u in range(num):
        xu = (1/w) * sum(sample[x]*np.exp(- 2j * np.pi*u*x/w)
                         for x in range(w))
        dft0.append(xu)

    A=np.array(dft0)
    A=A[np.newaxis,:]
    newcol=np.zeros((1,num))
    for i in range(1,n-w):
        diff=np.ones((1,num))*(sample[i+w]-sample[i])
        newcol= np.dot(A[i-1,:]+diff,V)
        A=np.vstack((A,newcol))
    
    B= np.zeros((A.shape[0],2*num))
    B[:, 0::2] = A.real
    B[:, 1::2] = A.imag
    return B


def MCB(sample, w, l, c, mean):
    windows = sliding_windows(sample, w)
    # A = [DFT(window, l, mean)for window in windows]
    # A = np.array(A)
    A=DFT_A(sample,w,l,mean)
    n, l = A.shape

    breakpoints = []

    for j in range(l):
        sorted_column = np.sort(A[:, j])
        bin_size = n // (c + 1)
        bin_indices = [i * bin_size for i in range(1, c + 1)]
        column_breakpoints = [sorted_column[i] for i in bin_indices]
        breakpoints.append(column_breakpoints)
    return breakpoints, A


def window_to_word(sample, w, l, c, mean):
    breakpoints, A = MCB(sample, w, l, c, mean)
    n, l = A.shape

    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    B_list = []

    for i in range(n):
        row_symbols = []
        for j in range(l):
            column_breakpoints = breakpoints[j]
            bin_index = np.digitize(A[i, j], column_breakpoints, right=True)
            bin_index = min(bin_index, c)
            row_symbols.append(alphabet[bin_index])
        B_list.append(''.join(row_symbols))
    B = np.array(B_list, dtype='<U101')
    return B


def plot_hist(n):
    fig, axs = plt.subplots(n, 2, figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n):
        sample = data_without_label.iloc[rd.randint(0,len(data_without_label))]  # first sample of length 1024
        sample = sample.tolist()
        boss = BOSSTransform(sample, w, l, c, mean)
        labels = list(boss.keys())
        values = list(boss.values())

        axs[i, 0].plot(sample, color=colors[i])
        axs[i, 0].set_xlabel('Temps')
        axs[i, 0].set_ylabel('Amplitude')
        axs[i, 0].set_title('Signal')

        axs[i, 1].bar(labels, values, color=colors[i])
        axs[i, 1].set_xlabel('Mot')
        axs[i, 1].set_ylabel('Fréquence')
        axs[i, 1].set_title('Histogramme des fréquences des mots')
        axs[i, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


plot_hist(4)