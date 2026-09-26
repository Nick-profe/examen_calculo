import numpy as np
 
# EXAMEN 2
# Cálculo aplicado a Machine Learning
 
# Nombre: Juan Esteban
# Apellido 1: Ruiz
# Apellido 2: Calero
# Rama: ruiz_calero
 
 
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
    return w*x + b
 
 
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
    return (
        f(x + h) - f(x)
    ) / h
 
# 4.1. PRUEBA DE DERIVADA NUMÉRICA
# ------------------------------------------------------------
derivative_at_3 = numerical_derivative(
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
 
    w = 0.0
    b = 0.0
 
    for epoch in range(epochs):
 
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
 
initial_prediction = predict(
    x,
    initial_w,
    initial_b
)
 
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
final_prediction = predict(
    x,
    w,
    b
)
 
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
x_new = 7
 
prediction = predict(
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
# Respuesta: w es la pendiente: indica cuánto aumenta el tiempo
# de preparación por cada producto adicional en el pedido.
# b es el intercepto: el tiempo estimado cuando x = 0, es decir,
# el tiempo base de preparación de un pedido.
 
 
# 2. ¿Qué representa la función de pérdida?
# Respuesta: El MSE mide qué tan alejadas están las predicciones
# de los valores reales (promedio de los errores al cuadrado).
# Entrenar el modelo significa reducir esta pérdida.
 
 
# 3. ¿A qué valor debería aproximarse la derivada
#    de x^2 en x = 3?
# Respuesta: A 6, porque f'(x) = 2x y f'(3) = 2(3) = 6.
 
 
# 4. ¿Qué representa dw?
# Respuesta: dw es la derivada parcial de la pérdida respecto a w
# (dL/dw). Indica cómo cambia la loss cuando se modifica w.
 
 
# 5. ¿Qué representa db?
# Respuesta: db es la derivada parcial de la pérdida respecto a b
# (dL/db). Indica cómo cambia la loss cuando se modifica b.
 
 
# 6. ¿Por qué Gradient Descent resta el gradiente
#    en lugar de sumarlo?
# Respuesta: Porque el gradiente apunta hacia la dirección de
# crecimiento de la función. Al restarlo nos movemos en la
# dirección contraria, lo que disminuye la loss.
 
 
# 7. ¿Qué significa que la pérdida final sea menor
#    que la pérdida inicial?
# Respuesta: Que el entrenamiento funcionó: los parámetros finales
# representan mejor los datos y las predicciones se acercan más
# a los tiempos reales que con el modelo inicial (w = 0, b = 0).
 
 
# 8. ¿Qué efecto tiene el learning rate
#    durante el entrenamiento?
# Respuesta: Controla el tamaño de cada actualización de los
# parámetros. Si es muy pequeño el entrenamiento es lento; si es
# muy grande puede volverse inestable y no converger.