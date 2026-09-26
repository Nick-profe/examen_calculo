import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Diego
# Apellido 1: Berrio
# Apellido 2: Lasso
# Rama: berrio_lasso


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
# Respuesta: w indica cuánto aumenta el tiempo de preparación por cada
# producto adicional en el pedido, y b sería el tiempo estimado si el
# pedido tuviera 0 productos.

# 2. ¿Qué representa la función de pérdida?
# Respuesta: Mide qué tan lejos están las predicciones del modelo respecto
# a los tiempos reales de preparación, elevando al cuadrado cada error y
# promediando.

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: A 6, ya que la derivada de x^2 es 2x, y 2*3 = 6.

# 4. ¿Qué representa dw?
# Respuesta: Es la derivada parcial de la pérdida respecto a w, indica en
# qué dirección y qué tanto hay que mover w para reducir el error.

# 5. ¿Qué representa db?
# Respuesta: Es la derivada parcial de la pérdida respecto a b, indica en
# qué dirección y qué tanto hay que mover b para reducir el error.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia donde la pérdida crece, así
# que para reducirla hay que moverse en la dirección contraria, restando.

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Que el modelo mejoró durante el entrenamiento, las
# predicciones quedaron más cerca de los valores reales que al principio.

# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: Controla qué tan grande es cada ajuste de w y b en cada paso;
# muy pequeño hace que el entrenamiento sea lento, muy grande puede hacer
# que nunca converja.