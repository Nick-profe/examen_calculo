import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: jeison
# Apellido 1: navarro
# Apellido 2: murillo
# Rama: navarro_murillo


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
    return x * w + b


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
    error = y_pred - y

    dw = (2 / len(x)) * np.sum(x * error)
    db = (2 / len(x)) * np.sum(error)

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
# Respuesta:w y b vienen siendo los parametros del modelo que se ajustan
# durante el entrenamiento, w nos muestra cuanto cambia el tiempo de preparacion por cada producto que se agregue en el pedido, osea la pendiente
# y b representa el tiempo base de preparacion cuando el pedido esta en 0 productos


# 2. ¿Qué representa la función de pérdida?
# Respuesta: la función de pérdida (mse_loss) mide qué tan mal está prediciendo el 
# modelo, comparando los valores reales (y) con las predicciones (y_pred).  entre más pequeño sea 
# su valor, mejor se ajusta el modelo a los datos. Es la cantidad que Gradient Descent 


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: Debería aproximarse a 6, ya que la derivada exacta de f(x) = x^2 es 
# f'(x) = 2x, y evaluada en x=3 da 2(3) = 6. En efecto, al correr el script, la 
# derivada numérica dio aproximadamente 6.000009999951316, muy cercana al valor exacto.


# 4. ¿Qué representa dw?
# Respuesta: dw es el gradiente de la función de pérdida respecto al parámetro w. 
# Indica en qué dirección y qué tan rápido cambia el error (mse_loss) si se modifica w. 
# Gradient Descent usa este valor para saber cuánto y hacia dónde ajustar w en cada paso 
# de entrenamiento, con el objetivo de reducir el error.


# 5. ¿Qué representa db?
# Respuesta: db es el gradiente de la función de pérdida respecto al parámetro b. 
# Indica en qué dirección y qué tan rápido cambia el error si se modifica b, y se usa 
# para ajustar b en cada paso de Gradient Descent.


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:  el gradiente apunta en la dirección donde la función de pérdida crece 
# más rápido. Como el objetivo es minimizar el error (no maximizarlo), Gradient 
# Descent debe moverse en la dirección opuesta al gradiente — por eso se resta en 
# lugar de sumarse. Sumar el gradiente empujaría los parámetros hacia donde el error 
# aumenta; restarlo los empuja hacia donde el error disminuye.


# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: significa que el modelo mejoró durante el entrenamiento: las predicciones 
# al final (con w y b ya ajustados) están más cerca de los valores reales que las 
# predicciones iniciales (con w=0, b=0). En este caso, la pérdida bajó de ~76.6 a 
# ~0.018, lo que indica que Gradient Descent encontró parámetros mucho mejores, y que 
# el modelo aprendió correctamente la relación entre número de productos y tiempo de 
# preparación.


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta:  el learning rate controla el tamaño del paso que da Gradient Descent en 
# cada actualización de w y b. Si es muy grande, el modelo puede dar pasos demasiado 
# bruscos, sobrepasar el mínimo del error, e incluso divergir (el error empeora en vez 
# de mejorar). Si es muy pequeño, el modelo converge de forma muy lenta, necesitando 
# muchas más épocas (epochs) para llegar a buenos valores de w y b. Encontrar un 
# learning rate adecuado es clave para que el entrenamiento sea eficiente y estable.