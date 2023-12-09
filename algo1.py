import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def BOSSTransform(sample, w, l, c, mean=False):
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
    n = len(sample)
    s = []
    for i in range(n-w):
        s.append(sample[i:i+w])
    return s


def DFT(sample, l, mean=False):
    n = len(sample)
    dft = []
    if mean:
        start = 0
    else:
        start = 1
    num = int(round(l/2))
    for u in range(start, num+start):
        xu = (1/n) * sum(sample[x]*np.exp(- 2j * np.pi*u*x/n)
                         for x in range(n))
        dft.append(xu.real)
        dft.append(xu.imag)
    return dft


def MCB(sample, w, l, c, mean):
    windows = sliding_windows(sample, w)
    A = [DFT(window, l, mean)for window in windows]
    A = np.array(A)
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
        sample = data_without_label.iloc[i]  # first sample of length 1024
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
