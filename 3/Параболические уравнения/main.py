# -*- coding: utf-8 -*-

#%% Либы + настройка

import numpy as np
import matplotlib.pyplot as plt
from num_methods import heat_eq_solve
from pandas import DataFrame

#%% Ввод параметров варианта

a = 2
b = 0
c = 1.25
f = lambda x, t: (x + 1) / (t + 4)
l = 4
h = 0.8
tau = 0.025
T = 0.1

N = int((6 - 2) / h + 1)
K = int(T / tau + 1)

phi0 = lambda t: -4 + t / (-1)
phil = lambda t: -39.41592654 + t ** 2 / (-2) - 0.4 * t

alpha = [[0, 1],
         [2, 2]]

psi = lambda x: -4 + 10 * np.sin(1.57079633 * x)

F = lambda x, t: f(x + 2, t)
Psi = lambda x: psi(x + 2)

#%% Решение

# N = 20
# K = 100

x, t, u = heat_eq_solve(a, Psi, phi0, phil, 
                        l, T,
                        N=N, K=K,
                        b=b, c=c, f=F,
                        alpha=alpha,
                        method="comb", order=1, tetta=0.5)

x = x + 2;
tau = t[1] - t[0]
h = x[1] - x[0]

u_table = DataFrame(u, columns=["t = {:.3f}".format(t_) for t_ in t ])

print(u_table)

#%% График при фиксированном времени

t_ = 0.05

ti = int(np.round(t_ * (K - 1) / T))

plt.figure(figsize=(7, 5), dpi=100) # 700x500
plt.plot(x, u[:, ti], "r", markersize = 3, label="Численное решение")
plt.xlim((x[0], x[-1])) 
plt.xlabel('x', fontsize=12, color='blue')
plt.ylabel('y', fontsize=12, color='blue')
plt.title("t = {:.3f}".format(t[ti]))
plt.legend()
plt.grid(True)
plt.savefig("plot.png", dpi=100)

