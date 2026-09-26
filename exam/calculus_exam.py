
# Examen 2
# Cálculo aplicado a Machine Learnin
# ============================================================

# Nombre: Jimmy 
# Apellido 1: Lopez
# Apellido 2: Yule
# Rama: lopez_yule

# Ejercicio

import numpy as np

#datos
x = np.array([1, 2, 3, 4, 5, 6])

y = np.array([
    5.0,
    7.1,
    9.0,
    11.2,
    12.9,
    15.1
])

# Prediction
def predict(x, w, b):
    return w*x + b

# w = 0
# b = 0

# y_pred = predict(x, w, b)

# print(y_pred)

# Perdida

def mse_loss(y_true, y_pred):
    return np.mean(
        (y_true - y_pred)**2
    )

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


# paso gradiente descendente
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

# Entrenamiento

# 1000 epocas

def train_linear_regression(
    x,
    y,
    learning_rate=0.01,
    epochs=1000
):

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


# MODELO INICIAL

initial_w = 0.0
initial_b = 0.0

# calcular la predicción inicial usando initial_w e initial_b

initial_prediction = predict(
    x,
    initial_w,
    initial_b
)

# calcular la pérdida inicial

initial_loss = mse_loss(
    y,
    initial_prediction
)


# ENTRENAMIENTO DEL MODELO

w, b = train_linear_regression(
    x,
    y,
    learning_rate=0.01,
    epochs=1000
)


# MODELO FINAL

# calcular las predicciones finales
final_prediction = predict(
    x,
    w,
    b
)

# calcular la pérdida final
final_loss = mse_loss(
    y,
    final_prediction
)


print("Pérdida inicial:", initial_loss)
print("Pérdida final:", final_loss)
print("Valor final de w:", w)
print("Valor final de b:", b)

print("prueba rate de aprendizaje:0.001")
w, b = train_linear_regression(
    x,
    y,
    learning_rate=0.001,
    epochs=1000
)

print("Valor w:", w)
print("Valor b:", b)


# NUEVA PREDICCIÓN


print("predecir x-new = 7")
# predecir para x_new = 7 usando los parámetros entrenados
x_new = 7

prediction = predict(
    x_new,
    w,
    b
)

print(
    "Predicción pedido con 7 productos:",
    prediction
)



# Preguntas

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta: w es la pendiente y b es el intercepto de la recta


# 2. ¿Qué representa la función de pérdida?
# Respuesta: Es el error de las predicciones respecto a los valores reales.


# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: Es 6


# 4. ¿Qué representa dw?
# Respuesta: Es la derivada de la función de pérdida con respecto a w


# 5. ¿Qué representa db?
# Respuesta: Es la derivada de la función de pérdida con respecto a b


# 6. ¿Por qué Gradient Descent resta el gradiente en lugar de sumarlo?
# Respuesta: Es el gradiente que esta en direccion del aumento de la pérdida, y al restarlo se busca que disminuya


# 7. ¿Qué significa que la pérdida final sea menor que la pérdida inicial?
# Respuesta: El modelo mejoró las predicciones y se esta mas cerca a los valoores reales


# 8. ¿Qué efecto tiene el learning rate durante el entrenamiento?
# Respuesta: Esto determina el tamaño de los pasos que se dan en la dirección del gradiente, afectando la velocidad de convergencia y la estabilidad del entrenamiento. Un learning rate demasiado alto puede causar que el modelo no converja, mientras que uno demasiado bajo puede hacer que el entrenamiento sea muy lento.


