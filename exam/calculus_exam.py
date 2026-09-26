import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Santiago
# Apellido 1: Gomez
# Apellido 2: Murcia
# Rama: gomez_murcia


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
# Respuesta: w es cuánto se demora de más el pedido por cada producto adicional, o sea la pendiente de la recta. b es el tiempo base que ya tiene el pedido aunque tenga pocos productos, como el tiempo fijo de alistarlo.


# 2. ¿Qué representa la función de pérdida?
# Respuesta: Es una forma de medir qué tan mal está prediciendo el modelo. Compara lo que predice con lo real, eleva la diferencia al cuadrado y saca el promedio. Entre más baja, mejor se ajusta la recta a los datos.


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: A 6, porque la derivada de x² es 2x y 2 por 3 da 6. En el código da algo como 6.00001 porque es una aproximación con h muy pequeño, no el valor exacto.


# 4. ¿Qué representa dw?
# Respuesta: Es cuánto cambia la pérdida cuando muevo un poquito w. Me dice hacia dónde y qué tanto debo ajustar la pendiente para bajar el error.


# 5. ¿Qué representa db?
# Respuesta: Lo mismo que dw pero para b: cuánto cambia la pérdida si muevo un poquito el intercepto, y en qué dirección tengo que corregirlo.


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia donde la pérdida sube más rápido, y lo que yo quiero es bajarla. Entonces me muevo en dirección contraria. Si lo sumara, el error iría aumentando en cada paso.


# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Que el modelo sí aprendió. Al inicio w y b son 0, así que predice 0 siempre y el error es enorme. Después del entrenamiento las predicciones quedan mucho más cerca de los tiempos reales.


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: Define qué tan grande es el paso que da el modelo en cada actualización. Si es muy pequeño, aprende muy lento. Si es muy grande, se pasa del mínimo y la pérdida puede oscilar o hasta dispararse. Con uno bien elegido, como 0.01 acá, la pérdida baja de forma estable.