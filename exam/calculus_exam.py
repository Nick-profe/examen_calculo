import numpy as np
from pathlib import Path

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Juan Camilo 
# Apellido 1: Henao 
# Apellido 2: Espinosa
# Rama: Henao_Espinosa

# 1. DATOS
# ------------------------------------------------------------

data_path = Path(__file__).resolve().parent.parent / "data" / "order_times.csv"
data = np.loadtxt(
    data_path,
    delimiter=",",
    skiprows=1
)

x = data[:, 0]
y = data[:, 1]

# 2. FUNCIÓN DE PREDICCIÓN
# ------------------------------------------------------------

def predict(x, w, b):
    return w * x + b

# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 4. DERIVADA NUMÉRICA
# ------------------------------------------------------------

def quadratic(x):
    return x**2

def numerical_derivative(
    f,
    x,
    h=1e-5
):
    return (f(x + h) - f(x - h)) / (2 * h)

# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
#usar numerical_derivative(quadratic, 3)
derivative_at_3 = numerical_derivative(quadratic, 3)

print(
    "Derivada aproximada de x^2 en x=3:",
    derivative_at_3
)

# 5. GRADIENTES DE LA REGRESIÓN LINEAL
# ------------------------------------------------------------

def linear_regression_gradients(
    x,
    y,
    w,
    b
):
    y_pred = predict(x, w, b)
    n = len(x)

    dw = (2 / n) * np.sum((y_pred - y) * x)
    db = (2 / n) * np.sum(y_pred - y)

    return dw, db

# 6. PASO DE GRADIENT DESCENT
# ------------------------------------------------------------

def gradient_descent_step(
    x,
    y,
    w,
    b,
    learning_rate
):
    dw, db = linear_regression_gradients(x, y, w, b)
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b

# 7. FUNCIÓN DE ENTRENAMIENTO
# ------------------------------------------------------------

def train_linear_regression(
    x,
    y,
    learning_rate=0.01,
    epochs=1000
):
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        w, b = gradient_descent_step(x, y, w, b, learning_rate)

    return w, b

# 8. MODELO INICIAL
# ------------------------------------------------------------

initial_w = 0.0
initial_b = 0.0

#calcular la predicción inicial usando initial_w e initial_b
initial_prediction = predict(x, initial_w, initial_b)

#calcular la pérdida inicial
initial_loss = mse_loss(y, initial_prediction)

# 9. ENTRENAMIENTO DEL MODELO
# ------------------------------------------------------------

w, b = train_linear_regression(
    x,
    y,
    learning_rate=0.01,
    epochs=1000
)

# 10. MODELO FINAL
# ------------------------------------------------------------
#calcular las predicciones finales
final_prediction = predict(x, w, b)

final_loss = mse_loss(y, final_prediction)

print("Pérdida inicial:", initial_loss)
print("Pérdida final:", final_loss)
print("Valor final de w:", w)
print("Valor final de b:", b)

# 11. NUEVA PREDICCIÓN
# ------------------------------------------------------------
#predecir para x_new = 7 usando los parámetros entrenados
x_new = 7

prediction = predict(x_new, w, b)

print(
    "Predicción para un pedido con 7 productos:",
    prediction
)

# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta:
# w es la pendiente de la recta, es decir, cuánto aumenta el tiempo de preparación
# por cada producto adicional. b es el término independiente, que representa el tiempo
# base cuando x = 0.

# 2. ¿Qué representa la función de pérdida?
# Respuesta:
# La función de pérdida mide qué tan lejos están las predicciones del modelo respecto a
# los valores reales. En este caso, usamos MSE, que promedia los errores al cuadrado.

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta:
# Debe aproximarse a 6, porque la derivada de x^2 es 2x y en x = 3 se obtiene 2*3 = 6.

# 4. ¿Qué representa dw?
# Respuesta:
# dw es la derivada parcial de la pérdida respecto a w. Indica cómo cambia la pérdida
# cuando se ajusta la pendiente w.

# 5. ¿Qué representa db?
# Respuesta:
# db es la derivada parcial de la pérdida respecto a b. Indica cómo cambia la pérdida
# cuando se ajusta el intercepto b.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:
# Porque el gradiente apunta en la dirección de mayor aumento de la pérdida. Para minimizar
# la pérdida, debemos movernos en la dirección contraria, es decir, restando el gradiente.

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta:
# Significa que el modelo ha aprendido a ajustar mejor la recta a los datos y que sus
# predicciones están más cerca de los valores reales.

# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta:
# El learning rate controla el tamaño del paso que da el algoritmo en cada actualización.
# Si es muy bajo, el entrenamiento es lento; si es muy alto, puede divergir o no converger.

