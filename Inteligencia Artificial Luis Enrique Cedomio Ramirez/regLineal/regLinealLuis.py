import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leer datos
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Calcular parámetros de mínimos cuadrados
N = len(X)
sumx = sum(X)
sumy = sum(Y)
sumxy = sum(X * Y)
sumx2 = sum(X * X)
w1 = (N * sumxy - sumx * sumy) / (N * sumx2 - sumx * sumx)
w0 = (sumy - w1 * sumx) / N
Ybar = w0 + w1 * X

# Configuración de la visualización
plt.figure(figsize=(12, 9))
plt.scatter(X, Y, color='blue', label='Datos')
plt.plot(X, Ybar, color='red', label='Mínimos Cuadrados')

# Etiquetas y título
plt.xlabel("X", fontsize=14)
plt.ylabel("Y", fontsize=14)
plt.title("Regresión Lineal: Ajuste por Mínimos Cuadrados", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
