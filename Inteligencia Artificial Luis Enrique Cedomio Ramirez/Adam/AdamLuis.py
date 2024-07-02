import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

# Leer datos
data = pd.read_csv('data.csv')
X = np.array(data.iloc[:, 0])
Y = np.array(data.iloc[:, 1])

# Mínimos cuadrados
N = len(X)
sumx = np.sum(X)
sumy = np.sum(Y)
sumxy = np.sum(X * Y)
sumx2 = np.sum(X * X)

# Parámetros
w = np.zeros(2, dtype=np.float32)
w[1] = (N * sumxy - sumx * sumy) / (N * sumx2 - sumx * sumx)
w[0] = (sumy - w[1] * sumx) / N
Ybar = w[0] + w[1] * X

# Descenso de gradiente (ADAM)
alpha = 0.1
epochs = 100

@jit(nopython=True)
def DG_ADAM(epochs, dim, sumx, sumy, sumxy, sumx2, N, alpha):
    error = np.zeros(epochs, dtype=np.float32)
    mn = np.zeros(dim, dtype=np.float32)
    vn = np.zeros(dim, dtype=np.float32)
    g = np.zeros(dim, dtype=np.float32)
    g2 = np.zeros(dim, dtype=np.float32)
    w = np.zeros(dim, dtype=np.float32)
    beta1 = 0.80
    beta2 = 0.999
    b1 = beta1
    b2 = beta2
    eps = 1.0e-8

    for i in range(epochs):
        g[0] = -2.0 * (sumy - w[0] * N - w[1] * sumx)
        g[1] = -2.0 * (sumxy - w[0] * sumx - w[1] * sumx2)
        g2 = g * g

        for j in range(dim):
            mn[j] = beta1 * mn[j] + (1.0 - beta1) * g[j]
            vn[j] = beta2 * vn[j] + (1.0 - beta2) * g2[j]

        b1 *= beta1
        b2 *= beta2
        mnn = mn / (1.0 - b1)
        vnn = vn / (1.0 - b2)
        fact = eps + np.sqrt(vnn)
        w -= (alpha / fact) * mnn

        Ybar2 = w[0] + w[1] * X
        error[i] = np.sum((Y - Ybar2) ** 2)

    return w, error

# Ejecutar el descenso de gradiente
w, error = DG_ADAM(epochs, 2, sumx, sumy, sumxy, sumx2, N, alpha)
print("Error = ", error[epochs - 1])
Ybar2 = w[0] + w[1] * X

# Gráfica
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Gráfico de dispersión y líneas de regresión
axes[0].scatter(X, Y, color='blue', label='Datos')
axes[0].plot(X, Ybar, color='red', label='Mínimos Cuadrados')
axes[0].plot(X, Ybar2, color='green', label='Descenso de Gradiente (ADAM)')
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("Regresión Lineal")
axes[0].legend()

# Gráfico de error
axes[1].plot(error, color='blue')
axes[1].set_ylabel("Error")
axes[1].set_xlabel("Épocas")
axes[1].set_title("Error durante el entrenamiento")

plt.tight_layout()
plt.show()
