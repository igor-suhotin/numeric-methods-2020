# -*- coding: utf-8 -*-

#%% Либы

import numpy as np
import matplotlib.pyplot as plt
from num_methods import wave_eq_solve
from pandas import DataFrame

#%% Ввод условий

a = 4
b = -1
c = -1
d = -4

f = lambda x, t: x * t / (1 + x ** 2 + 2 * t ** 2)
L = 3.1415926536
N = 5
T = 0.5
K = 6

phi0 = lambda t: -t / np.exp(t)
phil = lambda t: t ** 2 / np.exp(t)

alpha = [[0, -5], [0, -8]]

psi1 = lambda x: np.sin(x)
psi2 = lambda x: np.cos(x)

psi1_d = lambda x: np.cos(x)
psi1_dd = lambda x: -np.sin(x)

method = "explicit"

#%% Решение

x, t, u = wave_eq_solve(a, psi1, psi2, phi0, phil,
                        L, T,
                        N=N, K=K,
                        b=b, c=c, d=d, f=f,
                        alpha=alpha,
                        method=method, order1=1, order2=1)

h = x[1] - x[0]
tau = t[1] - t[0]

u_table = DataFrame(u, columns=["t = {:.3f}".format(t_) for t_ in t ])

print(u_table)

#%% График

t_ = 0.3

ti = int(np.round(t_ * (K - 1) / T))

plt.figure(figsize=(5, 4), dpi=100)
plt.plot(x, u[:, ti], "r", markersize = 1, label="Численное решение")
plt.xlim((x[0], x[-1]))
plt.xlabel('x', fontsize=12, color='blue')
plt.ylabel('y', fontsize=12, color='blue')
plt.title("t = {0}".format(t[ti]))
plt.legend()
plt.grid(True)
plt.savefig("plot.png")
