import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Wilson
# Apellido 1: Navia
# Apellido 2: Valencia
# Rama: navia_valencia


# 1. DATOS
# ------------------------------------------------------------

data = np.loadtxt(
    "data/order_times.csv",
    delimiter=",",
    skiprows=1
)

x = data[:, 0]
y = data[:, 1]


# 2. FUNCIÓN DE PREDICCIÓN
# ------------------------------------------------------------

def predict(x, w, b):
    # Calculamos la línea recta y_hat = w * x + b
    return w * x + b


# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    # Error Cuadrático Medio (promedio de los errores al cuadrado)
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n


# 4. DERIVADA NUMÉRICA
# ------------------------------------------------------------

def quadratic(x):
    return x**2


def numerical_derivative(
    f,
    x,
    h=1e-5
):
    # Fórmula de diferencias finitas centradas
    return (f(x + h) - f(x - h)) / (2 * h)

# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
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
    n = len(x)
    y_pred = predict(x, w, b)
    
    # Derivadas parciales del MSE respecto a w y b
    dw = (-2 / n) * np.sum((y - y_pred) * x)
    db = (-2 / n) * np.sum(y - y_pred)

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
    
    # Actualizamos w y b moviéndonos en contra del gradiente
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

    # Iteramos para ir ajustando los parámetros
    for _ in range(epochs):
        w, b = gradient_descent_step(x, y, w, b, learning_rate)

    return w, b


# 8. MODELO INICIAL
# ------------------------------------------------------------

initial_w = 0.0
initial_b = 0.0

initial_prediction = predict(x, initial_w, initial_b)
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
final_prediction = predict(x, w, b)
final_loss = mse_loss(y, final_prediction)


print("Pérdida inicial:", initial_loss)
print("Pérdida final:", final_loss)
print("Valor final de w:", w)
print("Valor final de b:", b)


# 11. NUEVA PREDICCIÓN
# ------------------------------------------------------------
x_new = 7
prediction = predict(x_new, w, b)

print(
    "Predicción para un pedido con 7 productos:",
    prediction
)


# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta: w representa la pendiente (cuánto aumenta el tiempo de preparación por cada producto extra) y b representa el intercepto (el tiempo base o fijo del pedido).

# 2. ¿Qué representa la función de pérdida?
# Respuesta: Mide qué tan equivocado está el modelo calculando el promedio de los errores al cuadrado entre los valores reales y las predicciones.

# 3. ¿A qué valor debería aproximarse la derivada de x^2 en x = 3?
# Respuesta: Debería aproximarse a 6, ya que analíticamente la derivada de x^2 es 2x, y evaluada en x=3 da 2 * 3 = 6.

# 4. ¿Qué representa dw?
# Respuesta: Es la derivada parcial del error con respecto a la pendiente w, indicando cómo afecta w al error total del modelo.

# 5. ¿Qué representa db?
# Respuesta: Es la derivada parcial del error con respecto al intercepto b, indicando cómo afecta b al error total.

# 6. ¿Por qué Gradient Descent resta el gradiente en lugar de sumarlo?
# Respuesta: Porque necesitamos movernos en dirección contraria al gradiente para descender hacia el valor mínimo de la función de pérdida (minimizar el error).

# 7. ¿Qué significa que la pérdida final sea menor que la pérdida inicial?
# Respuesta: Significa que el proceso de entrenamiento funcionó correctamente y el modelo ajustó sus parámetros para volverse más preciso.

# 8. ¿Qué efecto tiene el learning rate durante el entrenamiento?
# Respuesta: Controla el tamaño de los pasos que da el algoritmo al actualizar w y b. Si es muy grande puede hacer que el modelo falle al converger, y si es muy pequeño tardará demasiado en aprender.