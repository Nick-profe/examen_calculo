import numpy as np
from pathlib import Path

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre: Sebastian 
# Apellido 1: Rivas
# Apellido 2: Jimenez
# Rama: Rivas_Jimenez


# 1. DATOS
# ------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "order_times.csv"

data = np.loadtxt(
    DATA_FILE,
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
    return np.mean(
        (y_true - y_pred)**2
    )


# 4. DERIVADA NUMÉRICA
# ------------------------------------------------------------

def quadratic(x):
    return x**2


def numerical_derivative(
    f,
    x,
    h=1e-5
):
    return (f(x + h) - f(x) ) / h

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
    
    y_pred = predict(x, w, b)

    dw = (2/n) * np.sum(x * ( y - y_pred ))
    db = (2/n) * np.sum( y - y_pred )

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
        x, y, w, b)
    
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b


# 7. FUNCIÓN DE ENTRENAMIENTO
# ------------------------------------------------------------

def train_linear_regression(
    x,
    y,
    learning_rate=0.001,
    #learning_rate=0.01,
    epochs=1000
):

# TODO: entrenar el modelo y reemplazar estos valores

    w = 0.0
    b = 0.0
    
    for epoch in range(epochs):

        y_pred = predict(x,w,b)

        loss = mse_loss(y,y_pred)

        dw, db = linear_regression_gradients(x,y,w,b)

        w -= learning_rate*dw
        b -= learning_rate*db

        if epoch % 100 == 0:
            print(
                epoch,
                loss,
                w,b)

    return w, b

    w, b = train_linear_regression(x,y)

    print("w:", w)
    print("b:", b)


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
    learning_rate=0.001,
    #learning_rate=0.01,
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

prediction = predict( x_new, w, b )

print(
    "Predicción para un pedido con 7 productos:",
    prediction
)


# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta: w representa la pendiente de la linea de regresión, que indica el cambio del tiempo de entrega.
#               b representa el intercepto de la linea de regresió, que indica el punto cuando el tiempo de entrega es 0.


# 2. ¿Qué representa la función de pérdida?
# Respuesta: representa la diferencia entre el valor real y el valor predicho, asi podemos saber que tan bien esta funcionando nuestro modelo.



# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: deberia aproximarse a 6, ya que la derivada de x^2 es 2x.



# 4. ¿Qué representa dw?
# Respuesta: Estos representa el cambio  de la perdida con respecto a w.



# 5. ¿Qué representa db?
# Respuesta: Estos representa el cambio  de la perdida con respecto a b.


# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: El objetivo de la gradiente es apuntar hacia una dirección contraria de la función, de esta manera disminuimos la perdida y nos aproximamos al minimo.



# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Significa que el modelo esta aprendiendo a predecir mejor, ya que la pérdida se disminuye al final del entrenamiento.



# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: El learning rate define el tamaño de cada paso que toma el modelo durante el entramiento.