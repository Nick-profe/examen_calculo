import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Dairon
# Apellido 1: Rojas
# Apellido 2: Muñoz
# Rama: Rojas_Muñoz


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
    return np.mean((y_true - y_pred)**2)


# 4. DERIVADA NUMÉRICA
# ------------------------------------------------------------

def quadratic(x):
    return x**2


def numerical_derivative(
    f,
    x,
    h=1e-5
):
    return (f(x + h) - f(x))/h

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

    dw = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

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

    for epoch in range(epochs):
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
# Respuesta: w indica cuánto cambia el tiempo estimado por cada producto
# adicional. b es el tiempo que el modelo estimaría para cero productos.
# Tecnicamente, w es la pendiente y b es el punto de corte con el eje vertical.


# 2. ¿Qué representa la función de pérdida?
# Respuesta: Mide el error de las predicciones frente a los tiempos reales.
# Aquí usamos el promedio de esos errores al cuadrado (MSE).


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: A 6, porque la derivada de x^2 es 2x y, si x vale 3, da 6.


# 4. ¿Qué representa dw?
# Respuesta: Es el cambio de la pérdida al variar w. Tiene en cuenta
# el error de cada pedido y cuántos productos tenía. Si dw es negativo,
# aumentar la pendiente w ayuda a reducir la pérdida.


# 5. ¿Qué representa db?
# Respuesta: Es el cambio de la pérdida al variar b y este depende del error
# promedio. Si el modelo predice tiempos demasiado bajos, db es negativo
# y Gradient Descent aumenta b, moviendo toda la recta hacia arriba.


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia donde la pérdida aumenta.
# Al restarlo, movemos los parámetros hacia una pérdida menor.


# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Que, para estos datos, el modelo entrenado se equivoca
# menos que cuando empezamos con w y b en cero.


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: Define el tamaño de cada ajuste de w y b. Si es muy pequeño,
# se aprende lento; si es muy grande, puede saltarse el mínimo y volverse inestable.