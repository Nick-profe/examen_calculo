import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Sebastián 
# Apellido 1: Moreno
# Apellido 2: Dorado
# Rama: moreno_dorado


# 1. DATOS
# ------------------------------------------------------------

data = np.loadtxt(
    "examen_calculo/data/order_times.csv",
    delimiter=",",
    skiprows=1
)

x = data[:, 0]
y = data[:, 1]


# 2. FUNCIÓN DE PREDICCIÓN
# ------------------------------------------------------------

def predict(x, w, b):
    # Completar
    return w * x + b
    pass


# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    # Completar
    return np.mean((y_true - y_pred) ** 2)
    pass


# 4. DERIVADA NUMÉRICA
# ------------------------------------------------------------

def quadratic(x):
    return x**2


def numerical_derivative(
    f,
    x,
    h=1e-5
):
    # Completar
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

    n = len(x)

    y_pred = predict(
        x,
        w,
        b
    )

    dw = (
        (-2/n)
        * np.sum(
            x * (y - y_pred)
        )
    )

    db = (
        (-2/n)
        * np.sum(
            y - y_pred
        )
    )

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

    dw, db = linear_regression_gradients(
        x,
        y,
        w,
        b
    )

    w = w - learning_rate*dw
    b = b - learning_rate*db

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
# Respuesta:

#W representa la pendiente de la línea de regresión, es decir, cuánto cambia el tiempo de entrega (y) por cada unidad adicional de productos (x).
#B representa la intersección con el eje y, es decir, el tiempo de entrega estimado cuando no hay productos (x=0).

# 2. ¿Qué representa la función de pérdida?
# Respuesta:

#La función de pérdida representa la medida de error entre las predicciones del modelo y los valores reales. 
#En este caso, se utiliza el error cuadrático medio (MSE) para cuantificar qué tan lejos están las predicciones de los tiempos de entrega reales.

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta:

#La derivada de x^2 en x = 3 debería aproximarse a 6, ya que la derivada de x^2 es 2x, y al evaluarla en x = 3 obtenemos 2*3 = 6.

# 4. ¿Qué representa dw?
# Respuesta:

#Dw representa la derivada parcial de la función de pérdida con respecto al parámetro w (pendiente).
#Indica cómo cambia la pérdida cuando se ajusta w, y se utiliza para actualizar w durante el proceso de entrenamiento mediante el descenso de gradiente.

# 5. ¿Qué representa db?
# Respuesta:

#Db representa la derivada parcial de la función de pérdida con respecto al parámetro b (intersección).
#Indica cómo cambia la pérdida cuando se ajusta b, y se utiliza para actualizar b durante el proceso de entrenamiento mediante el descenso de gradiente.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:

#Gradient Descent resta el gradiente porque el gradiente indica la dirección de mayor aumento de la función de pérdida.

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta:

#Que el modelo ha aprendido a ajustar sus parámetros (w y b) de manera que las predicciones se acercan más a los valores reales, reduciendo así el error y mejorando la precisión del modelo.

# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta:

#El learning rate determina el tamaño de los pasos que se toman en la dirección del gradiente durante el entrenamiento.