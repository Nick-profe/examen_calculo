import numpy as np

# EXAMEN 2
# Cálculo aplicado a Machine Learning

# Nombre:Jaiber
# Apellido 1: Obando
# Apellido 2: Lopez
# Rama: obando_lopez


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
     return w * x + b
pass


# 3. FUNCIÓN DE PÉRDIDA
# ------------------------------------------------------------

def mse_loss(y_true, y_pred):
    # Completar
    return np.mean((
        y_true - y_pred) ** 2
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
    return (f(x + h) - f(x)) / h
    pass

# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
# TODO: usar numerical_derivative(quadratic, 3)
#derivative_at_3 = None
derivative_at_3 = numerical_derivative(quadratic,3)

print("Derivada numérica en x=3:", derivative_at_3)

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
    y_pred = predict(x, w, b)
    dw = -(2/n) * np.sum(x * (y - y_pred))
    db = -(2/n) * np.sum(y - y_pred)
    #dw = None
    #db = None

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
    dw, db = linear_regression_gradients(x, y, w, b)
    w_new = w - learning_rate * dw
    b_new = b - learning_rate * db
    return w_new, b_new


# 7. FUNCIÓN DE ENTRENAMIENTO
# ------------------------------------------------------------

def train_linear_regression(
    x,
    y,
    learning_rate=0.01,
    epochs=1000
):

# TODO: entrenar el modelo y reemplazar estos valores

    #w = None
    #b = None

    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        y_pred = predict(x, w, b)
        loss=mse_loss(y,y_pred)
        dw,db=linear_regression_gradients(x,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db

    if epoch % 100==0:
        print(epoch,loss,w,b)

    return w, b

w,b=train_linear_regression(x,y)
print("w:",w)
print("b:",b)

# 8. MODELO INICIAL
# ------------------------------------------------------------

initial_w = 0.0
initial_b = 0.0

# TODO: calcular la predicción inicial usando initial_w e initial_b
#initial_prediction = None
initial_prediction=predict(x,initial_w,initial_b)


# TODO: calcular la pérdida inicial
#initial_loss = None
initial_loss=mse_loss(y,initial_prediction)

final_prediction= predict(x,w,b)
final_loss=mse_loss(y,final_prediction)

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
#final_prediction = None
final_prediction= predict(x,w,b)

# TODO: calcular la pérdida final
#final_loss = None
final_loss=mse_loss(y,final_prediction)

print("Pérdida inicial:", initial_loss)
print("Pérdida final:", final_loss)
print("Valor final de w:", w)
print("Valor final de b:", b)


# 11. NUEVA PREDICCIÓN
# ------------------------------------------------------------
# TODO: predecir para x_new = 7 usando los parámetros entrenados
x_new = 7

#prediction = None
prediction=predict(x_new,w,b)


print(
    "Predicción para un pedido con 7 productos:",
    prediction
)


# 12. PREGUNTAS DE INTERPRETACIÓN
# ------------------------------------------------------------

# 1. ¿Qué representan w y b dentro de este problema?
# Respuesta:
# w representa cuánto cambia la predicción [tiempo de preparación de un pedido]
# cuanto x [cantidad de productos] aumenta en una unidad
# b representa el valor estimado cuando x[cantidad de productos] es 0

# 2. ¿Qué representa la función de pérdida?
# Respuesta:
# La función de perdida representa que tan lejos están las predicciones de los
# valores reales, en el caso particular planteado que tan lejos está la predicción de tiempos
# de los valores de tiempo reales [preparation_time]

# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: Debería aproximarse a 6


# 4. ¿Qué representa dw?
# Respuesta:
#dw = Derivada parcial de la pérdida (loss) con respecto a w.


# 5. ¿Qué representa db?
# Respuesta:
#db = Derivada parcial de la pérdida (loss) con respecto a b.

# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta:
#Porque el gradiente apunta hacia una dirección de crecimiento de la función. La direccción contraria ayuda a disminuir la loss

# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta:
# Que los parámetros finales representan mejor los datos.

# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta:
#Un learning rate demasiado pequeño puede hacer lento el modelo, un learning rate muy grande puede volver el entreamiento inestable