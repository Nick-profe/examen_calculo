import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Andres
# Apellido 1: Pinilla
# Apellido 2: Victoria
# Rama: pinilla_victoria


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
# Respuesta: w muestra qué tanto crece el tiempo de preparación por cada
# producto que se le añade al pedido, mientras que b sería el tiempo que
# tomaría un pedido hipotético de cero productos.

# 2. ¿Qué representa la función de pérdida?
# Respuesta: Es una forma de medir qué tan equivocado está el modelo,
# comparando lo que predijo contra los tiempos reales y penalizando más
# los errores grandes al elevarlos al cuadrado.

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: Debería dar cerca de 6, porque la derivada de x^2 es 2x, y
# evaluada en 3 da 2(3) = 6.

# 4. ¿Qué representa dw?
# Respuesta: Es qué tan sensible es el error frente a cambios en w, y le
# dice al algoritmo hacia qué lado moverlo para que el error baje.

# 5. ¿Qué representa db?
# Respuesta: Lo mismo que dw pero para el parámetro b, indica el ajuste
# necesario en el intercepto para reducir el error.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: El gradiente señala hacia donde el error aumenta, entonces
# para bajarlo hay que ir en sentido contrario, y por eso se resta.

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Significa que el entrenamiento sirvió, el modelo terminó
# prediciendo mucho más parecido a los datos reales que al comienzo.

# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: Define el tamaño de cada paso que da el modelo al actualizar
# sus parámetros; si es muy chico el entrenamiento avanza lentísimo, y si
# es muy grande puede que nunca se estabilice.