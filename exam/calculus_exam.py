import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre:
# Apellido 1: Monsalve
# Apellido 2: Gomez
# Nombre : Leonardo
# Rama: Monsalve_Gomez


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
    # Modelo lineal: y_hat = w*x + b
    return w * x + b


# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    # MSE = (1/n) * sum((y_i - y_hat_i)^2)
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
    # Aproximación: f'(x) ≈ (f(x + h) - f(x)) / h
    return (f(x + h) - f(x)) / h

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
    error = y - y_pred

    # dL/dw = -(2/n) * sum(x_i * (y_i - y_hat_i))
    dw = (-2 / n) * np.sum(x * error)

    # dL/db = -(2/n) * sum(y_i - y_hat_i)
    db = (-2 / n) * np.sum(error)

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

    # Se resta el gradiente para moverse en la dirección
    # en la que la pérdida disminuye
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
    # Parámetros iniciales
    w = 0.0
    b = 0.0

    for epoch in range(epochs):

        y_pred = predict(x, w, b)
        loss = mse_loss(y, y_pred)

        w, b = gradient_descent_step(
            x,
            y,
            w,
            b,
            learning_rate
        )

        if epoch % 100 == 0:
            print(
                "Epoch:", epoch,
                "| Loss:", loss,
                "| w:", w,
                "| b:", b
            )

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
# Respuesta:

"""
w es la pendiente: indica cuántos minutos aumenta el tiempo de preparación estimado por cada producto adicional
en el pedido. 
b es el intercepto: el tiempo estimado cuando x = 0, que se puede interpretar como un tiempo base o fijo
de preparación (alistar el pedido, empacar, etc.).
Con los datos del ejercicio se obtuvo w ≈ 1.99 y b ≈ 1.09: cada producto adicional agrega cerca de 2 minutos y hay un
tiempo base de aproximadamente 1.1 minutos.

"""


# 2. ¿Qué representa la función de pérdida?
# Respuesta:

"""
Mide qué tan lejos están las predicciones del modelo de los tiempos reales. 
El MSE promedia los errores al cuadrado, por lo que siempre es positivo y castiga más los errores grandes. 
Entre menor sea, mejor se ajusta el modelo a los datos.

"""

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta:


# 4. ¿Qué representa dw?
# Respuesta:


# 5. ¿Qué representa db?
# Respuesta:


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:


# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta:


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: