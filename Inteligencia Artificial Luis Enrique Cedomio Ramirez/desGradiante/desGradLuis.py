import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

# Leer datos
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Mínimos cuadrados
N = len(X)
sumx = sum(X)
sumy = sum(Y)
sumxy = sum(X * Y)
sumx2 = sum(X * X)
w1 = (N * sumxy - sumx * sumy) / (N * sumx2 - sumx * sumx)
w0 = (sumy - w1 * sumx) / N
Ybar = w0 + w1 * X

# Inicialización de los parámetros para descenso de gradiente
w0 = 0.0
w1 = 0.0
alpha = 0.025
epochs = 100

# Definición de la función de descenso de gradiente
@jit(nopython=True)
def descensoG(epochs, sumx, sumy, sumxy, sumx2, N, alpha):
    w0 = 0.0
    w1 = 0.0
    for i in range(epochs):
        Gradw0 = -2.0 * (sumy - w0 * N - w1 * sumx)
        Gradw1 = -2.0 * (sumxy - w0 * sumx - w1 * sumx2)
        w0 -= alpha * Gradw0
        w1 -= alpha * Gradw1
    return w0, w1

# Ejecutar el descenso de gradiente
w0, w1 = descensoG(epochs, sumx, sumy, sumxy, sumx2, N, alpha)
Ybar2 = w0 + w1 * X

# Configuración de la visualización
plt.figure(figsize=(12, 9))
plt.scatter(X, Y, color='blue', label='Datos')
plt.plot(X, Ybar, color='red', label='Mínimos Cuadrados')
plt.plot(X, Ybar2, color='green', label='Descenso de Gradiente')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regresión Lineal: Mínimos Cuadrados vs. Descenso de Gradiente")
plt.legend()
plt.grid(True)
plt.show()
