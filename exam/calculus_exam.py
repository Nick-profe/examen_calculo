import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Santiago
# Apellido 1: Cuellar
# Apellido 2: Andrade
# Rama: Cuellar_Andrade


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
    n = x.shape[0]
    y_pred = predict(x, w, b)
    error = y_pred - y

    dw = (2 / n) * np.sum(error * x)
    db = (2 / n) * np.sum(error)

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
# Respuesta: es que w es la pendiente quiere decir que es cuando aumenta el tiempo 
# de preparación de un pedido por cada producto que llegue o contiene y el intercepto b
# es el tiempo base de preparacion cuando tiene 0 productos.
#

# 2. ¿Qué representa la función de pérdida?
# Respuesta: Mide si la prediccion (y_pred) esta muy diferente a la
# de los valores reales (y_true), usando el promedio al cuadrado 
# (y_true - y_pred)**2

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: la derivada analítica de x^2 es 2x, y 2*3 = 6.

# 4. ¿Qué representa dw?
# Respuesta: Es el gradiente de la pérdida respecto a w; indica la
#  magnitud en que hay que cambiar w para reducir el loss.

# 5. ¿Qué representa db?
# Respuesta: Es el gradiente de la pérdida respecto a b; indica cómo
# ajustar el intercepto para reducir el loss del modelo.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia una dirección de 
# crecimiento de la función. La dirección contraria ayuda a disminuir la loss.

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Significa que el modelo aprendió de sus predicciones al
# final del entrenamiento se acercan más a los valores reales que
# las predicciones hechas con los parámetros iniciales (w=0, b=0)
# y tambien que el loss es muy minimo .

# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: Controla el tamaño del paso en cada actualización de w y b.
# Un valor muy pequeño hace que el entrenamiento sea lento; uno muy
# grande puede hacer que el modelo puede volverse inestable.