import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Eduardo
# Apellido 1:Sarmiento
# Apellido 2:Pradilla
# Rama:sarmiento_pradilla


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
    pass


# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    # Completar
    return np.mean(
        (y_true - y_pred)**2
    )
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
    return (
        f(x + h) - f(x)
    ) / h
    pass

# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
# TODO: usar numerical_derivative(quadratic, 3)
derivative_at_3 =  numerical_derivative(
    quadratic,
    3
)


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
    # Completar
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
    # Completar
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
final_loss =  mse_loss(
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

prediction =  predict(
    x_new,
    w,
    b
)


print(
    "Predicción para un pedido con 7 productos:",
    prediction
)


# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta:

# w es la pendiente y b el intercepto.

# 2. ¿Qué representa la función de pérdida?
# Respuesta:

# Mide el error de las predicciones respecto a los valores reales.

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta:

# Debe aproximarse a 6.


# 4. ¿Qué representa dw?
# Respuesta:

# dw representa la derivada parcial de la pérdida respecto al parámetro w: ∂L/∂w.


# 5. ¿Qué representa db?
# Respuesta:

# db representa la derivada parcial de la pérdida respecto al parámetro b: ∂L/∂b.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:

# Porque buscamos movernos en una dirección que reduzca la loss.


# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta:

#Que las predicciones se acercan más a los valores reales.


# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta:

# Controla el tamaño de cada actualización de los parámetros.