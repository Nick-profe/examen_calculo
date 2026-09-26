# examen_calculo
Repositorio exclusivo para el desarrollo del examen de Cálculo del curso de Herramientas Matemáticas y Computacionales para la IA

## Objetivo

Implementar un modelo de regresión lineal entrenado mediante Gradient Descent utilizando únicamente NumPy.

## Caso

Una empresa desea predecir el tiempo de preparación de un pedido según el número de productos que contiene.

El modelo es:

$$
\hat{y}=wx+b
$$

El estudiante deberá implementar las funciones necesarias para calcular predicciones, MSE, derivadas, gradientes y entrenar el modelo mediante Gradient Descent.

## Reglas

1. Trabaje únicamente en su rama individual.
2. El nombre de la rama debe ser:

   `apellido1_apellido2`

3. No realizar cambios sobre `main`.
4. No realizar cambios sobre `examen_2`.
5. No crear Pull Request.
6. No realizar merge.
7. Solo se permite NumPy.
8. No se permite utilizar scikit-learn, scipy.optimize ni funciones que entrenen automáticamente el modelo.
9. La entrega corresponde al último commit disponible en su rama remota al cierre del examen.

## Archivos a modificar

Únicamente:

`exam/calculus_exam.py`

## Entrega

```bash
git add exam/calculus_exam.py
git commit -m "Complete calculus exam"
git push -u origin apellido1_apellido2