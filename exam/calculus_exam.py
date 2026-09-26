import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Maira Alejandra
# Apellido 1: Balanta
# Apellido 2: Peña
# Rama: balanta_pena


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
    return (f(x + h) - f(x - h)) / (2 * h)

# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
# TODO: usar numerical_derivative(quadratic, 3)
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

    dw = np.mean(2 * x * (y_pred - y))
    db = np.mean(2 * (y_pred - y))

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

# TODO: entrenar el modelo y reemplazar estos valores

    w = 0.0
    b = 0.0

    for _ in range(epochs):
        w, b = gradient_descent_step(
            x,
            y,
            w,
            b,
            learning_rate
        )

    return w, b


# 8. MODELO INICIAL
# ------------------------------------------------------------

initial_w = 0.0
initial_b = 0.0

# TODO: calcular la predicción inicial usando initial_w e initial_b
initial_prediction = predict(x, initial_w, initial_b)

# TODO: calcular la pérdida inicial
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
# TODO: calcular las predicciones finales
final_prediction = predict(x, w, b)

# TODO: calcular la pérdida final
final_loss = mse_loss(y, final_prediction)


print("Pérdida inicial:", initial_loss)
print("Pérdida final:", final_loss)
print("Valor final de w:", w)
print("Valor final de b:", b)


# 11. NUEVA PREDICCIÓN
# ------------------------------------------------------------
# TODO: predecir para x_new = 7 usando los parámetros entrenados
x_new = 7

prediction = predict(x_new, w, b)

print(
    "Predicción para un pedido con 7 productos:",
    prediction
)


# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta: w representa cuánto aumenta aproximadamente el tiempo
# de preparación por cada producto adicional. b representa el tiempo
# estimado de preparación cuando el pedido tiene 0 productos.


# 2. ¿Qué representa la función de pérdida?
# Respuesta: La función de pérdida mide qué tan alejadas están las
# predicciones del modelo respecto a los valores reales. En este caso
# se utiliza el error cuadrático medio (MSE).


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: Debería aproximarse a 6, porque la derivada de x^2 es 2x
# y al evaluarla en x = 3 se obtiene 6.


# 4. ¿Qué representa dw?
# Respuesta: dw representa la derivada de la función de pérdida respecto
# a w. Indica cómo cambia la pérdida cuando cambia el valor de w.


# 5. ¿Qué representa db?
# Respuesta: db representa la derivada de la función de pérdida respecto
# a b. Indica cómo cambia la pérdida cuando cambia el valor de b.


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:  Porque el gradiente indica la dirección en la que la pérdida
# aumenta. Al restarlo, los parámetros se actualizan en la dirección
# que busca disminuir la pérdida.



# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Significa que el modelo aprendió parámetros que producen
# predicciones más cercanas a los valores reales del conjunto de datos.



# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: El learning rate determina el tamaño de los pasos que se
# realizan al actualizar los parámetros. Un valor muy pequeño hace que
# el aprendizaje sea lento, mientras que uno muy grande puede provocar
# que el modelo no llegue adecuadamente al mínimo de la pérdida.
