import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Johan 
# Apellido 1: Jurado
# Apellido 2: Lloreda
# Rama:


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
    # Completar
    return w*x + b


# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    # Completar
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

    for epoch in range(epochs):

        y_pred = predict(
            x,
            w,
            b
        )

        loss = mse_loss(
            y,
            y_pred
        )

        dw, db = linear_regression_gradients(
            x,
            y,
            w,
            b
        )

        w -= learning_rate*dw
        b -= learning_rate*db

        if epoch % 100 == 0:
            print(
                epoch,
                loss,
                w,
                b
            )

    return w, b

# 8. MODELO INICIAL
# ------------------------------------------------------------

initial_w = 0.0
initial_b = 0.0

# TODO: calcular la predicción inicial usando initial_w e initial_b
initial_prediction = predict(
    x,
    initial_w,
    initial_b
)

# TODO: calcular la pérdida inicial
initial_loss = mse_loss(
    y,
    initial_prediction
)

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
final_prediction = predict(
    x,
    w,
    b
)

# TODO: calcular la pérdida final
final_loss = mse_loss(
    y,
    final_prediction
)


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
# Respuesta: W representa cuánto tiempo adicional se necesita para preparar cada producto que se agrega al pedido. B representa el tiempo base de preparación.


# 2. ¿Qué representa la función de pérdida?
# Respuesta: La funcion de perdida mide qué tan alejadas están las predicciones de los valores reales.


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: El valor al que se aproxima es a 6.0


# 4. ¿Qué representa dw?
# Respuesta: Es el gradiente que Indica cómo cambia la loss respecto a w


# 5. ¿Qué representa db?
# Respuesta: Es el gradiente que Indica cómo cambia la loss respecto a b


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia una dirección de crecimiento de la función. La dirección contraria ayuda a disminuir la loss.


# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Significa que los parámetros finales representan mejor los datos.


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: El learning rate controla el tamaño de la actualización. demasiado pequeño puede hacer lento el entrenamiento. Uno demasiado grande puede volverlo inestable.