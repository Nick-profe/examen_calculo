import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Juan David
# Apellido 1: Velasco
# Apellido 2: Otero
# Rama: velasco_otero


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
    return (f(x + h) - f(x)) / h

# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
#: usar numerical_derivative(quadratic, 3)
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

    dw = (-2 / n) * np.sum(x * (y - y_pred))
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

#: entrenar el modelo y reemplazar estos valores

    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        w, b = gradient_descent_step(x, y, w, b, learning_rate)

    return w, b


# 8. MODELO INICIAL
# ------------------------------------------------------------

initial_w = 0.0
initial_b = 0.0

#: calcular la predicción inicial usando initial_w e initial_b
initial_prediction = predict(x, initial_w, initial_b)

#: calcular la pérdida inicial
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
#: calcular las predicciones finales
final_prediction = predict(x, w, b)

#: calcular la pérdida final
final_loss = mse_loss(y, final_prediction)


print("Pérdida inicial:", initial_loss)
print("Pérdida final:", final_loss)
print("Valor final de w:", w)
print("Valor final de b:", b)


# 11. NUEVA PREDICCIÓN
# ------------------------------------------------------------
# predecir para x_new = 7 usando los parámetros entrenados
x_new = 7

prediction = predict(x_new, w, b)

print(
    "Predicción para un pedido con 7 productos:",
    prediction
)


# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta: w es la pendiente e indica cuánto aumenta el tiempo estimado por cada
# producto adicional. b es el intercepto y representa el tiempo base estimado cuando no hay productos.


# 2. ¿Qué representa la función de pérdida?
# Respuesta: Mide el error cuadrático medio (MSE) entre los tiempos reales
# de preparación y las predicciones realizadas por el modelo.


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: 6


# 4. ¿Qué representa dw?
# Respuesta: Representa la derivada parcial de la pérdida respecto a w (dL/dw),
# indicando la dirección y magnitud de cambio en la pérdida al modificar la pendiente.


# 5. ¿Qué representa db?
# Respuesta: Representa la derivada parcial de la pérdida respecto a b (dL/db),
#  indicando cómo cambia la pérdida al modificar el intercepto.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia la dirección de máximo crecimiento de la función de pérdida.
# Restarlo permite avanzar en dirección opuesta para minimizar la pérdida.

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Significa que las predicciones del modelo ajustado se han acercado a los datos reales y que el proceso de entrenamiento fue exitoso.


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: El learning rate determina el tamaño de los pasos que se toman en cada iteración del algoritmo
# de optimización. Un learning rate demasiado alto puede hacer que el modelo no converja,
# mientras que uno demasiado bajo puede hacer que el entrenamiento sea muy lento.