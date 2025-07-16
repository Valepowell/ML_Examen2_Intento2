# ML_Examen2_Intento2
# Análisis Predictivo de Suscripción a Depósitos a Plazo

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar modelos de clasificación para predecir si un cliente de un banco suscribirá un depósito a plazo, basándose en datos de campañas de marketing. El análisis se enfoca en la limpieza de datos, la exploración de las características de los clientes y la evaluación comparativa de diferentes algoritmos de clasificación para identificar el más adecuado para esta tarea, considerando el desbalance inherente en la variable objetivo.

## Datos

El análisis se realizó utilizando el dataset `bank-full.csv`, que contiene información sobre clientes de un banco y resultados de campañas de marketing directo. El dataset original se obtuvo de [indicar fuente si es conocida, por ejemplo, un repositorio público o Kaggle].

El dataset incluye variables como:
- Características demográficas del cliente (edad, trabajo, estado civil, educación).
- Información financiera (balance, si tiene default, hipoteca, préstamo personal).
- Información de la última campaña de contacto (contacto, día, mes, duración).
- Información de campañas anteriores (días desde el último contacto, número de contactos previos, resultado de la campaña anterior).
- Variable objetivo: `y` (si el cliente suscribió un depósito a plazo - 'yes' o 'no').

## Metodología

El proyecto siguió las siguientes etapas:

1.  **Carga y Limpieza de Datos:**
    - Carga del dataset `bank-full.csv`.
    - Estandarización de nombres de columnas.
    - Verificación de duplicados y valores nulos.
    - Identificación y manejo de valores 'unknown' en columnas categóricas.

2.  **Exploración de Datos (EDA):**
    - Estadísticas descriptivas para variables numéricas y categóricas.
    - Visualizaciones univariadas (histogramas, boxplots) para entender la distribución de las características.
    - Análisis de correlación entre variables numéricas.
    - Visualización de la relación entre variables categóricas y la variable objetivo `y` (gráficos de barras apiladas).
    - Identificación del desbalance de clases en la variable objetivo `y`.

3.  **Preprocesamiento para Modelado:**
    - Separación de características (X) y variable objetivo (y).
    - División del dataset en conjuntos de entrenamiento y prueba (80/20).
    - Creación de un pipeline de preprocesamiento utilizando `ColumnTransformer` para:
        - Escalar variables numéricas (`StandardScaler`).
        - Codificar variables categóricas (`OneHotEncoder`), incluyendo la categoría 'unknown'.

4.  **Implementación y Evaluación de Modelos de Clasificación:**
    - Implementación de pipelines que combinan el preprocesador con diferentes modelos de clasificación:
        - Regresión Logística
        - Árbol de Decisión (`DecisionTreeClassifier`)
        - Random Forest (`RandomForestClassifier`)
        - Support Vector Machine (`SVC`)
    - Para mitigar el desbalance de clases, se utilizó el parámetro `class_weight='balanced'` en los clasificadores.
    - Evaluación de cada modelo en el conjunto de prueba utilizando métricas clave para datasets desbalanceados: Accuracy, Precision ('yes'), Recall ('yes'), F1-Score ('yes'), y Matriz de Confusión.

5.  **Optimización de Hiperparámetros (Random Forest):**
    - Realización de una búsqueda de hiperparámetros utilizando `GridSearchCV` en el pipeline del Random Forest, optimizando el F1-Score para la clase 'yes'. Se utilizó una cuadrícula de parámetros reducida debido a restricciones de tiempo.

## Resultados Clave

La comparación del rendimiento de los modelos en el conjunto de prueba (con `class_weight='balanced'`) mostró lo siguiente para la clase minoritaria ('yes' - Suscripción):

| Modelo                | Precision ('yes') | Recall ('yes') | F1-Score ('yes') | Accuracy |
| :-------------------- | :---------------- | :------------- | :--------------- | :------- |
| Regresión Logística   | 0.4224            | **0.8304**     | 0.5600           | 0.8425   |
| Árbol de Decisión     | 0.4740            | 0.4170         | 0.4437           | 0.8738   |
| Random Forest (Base)  | **0.6875**        | 0.3327         | 0.4484           | **0.9012**|
| SVM                   | 0.4291            | **0.8900**     | **0.5790**       | 0.8439   |
| Random Forest (Optimizado) | 0.4193 | 0.8570 | 0.5631 | 0.8395 |

## Conclusiones

- El desbalance de clases es un factor crítico que afecta el rendimiento de los modelos.
- El modelo **SVM** logró el mejor **F1-Score** para la clase 'yes' y la mayor **Exhaustividad (Recall)**, siendo el más efectivo en identificar a la mayoría de los clientes que suscribieron.
- El **Random Forest** base tuvo la mayor **Precisión** en la clase 'yes', siendo el más confiable en sus predicciones positivas, aunque con bajo Recall.
- La optimización de hiperparámetros con la cuadrícula reducida no mejoró significativamente el rendimiento del Random Forest en las métricas de la clase minoritaria en comparación con el SVM base.

La elección del mejor modelo depende de la prioridad del negocio: si es identificar a la mayor cantidad de potenciales suscriptores (Recall), SVM o Regresión Logística son preferibles; si es asegurar que las predicciones positivas sean correctas (Precision), Random Forest es mejor. El SVM ofrece el mejor equilibrio según el F1-Score.

## Cómo Ejecutar el Código

Este análisis se realizó en un cuaderno de Google Colab. Para ejecutar el código:

1.  Asegúrate de tener acceso al dataset `bank-full.csv` y cargarlo correctamente (por ejemplo, montando Google Drive).
2.  Ejecuta las celdas del cuaderno secuencialmente.
3.  Las bibliotecas necesarias son `pandas`, `numpy`, `matplotlib`, `seaborn`, y módulos de `sklearn` (`train_test_split`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, clasificadores como `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `GridSearchCV`, y métricas de `sklearn.metrics`). Asegúrate de que estén instaladas en tu entorno.
