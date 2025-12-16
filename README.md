Proyecto Predicción de Ganador en Partidas de League of Legends
Descripción

Este proyecto tiene como objetivo predecir el ganador de una partida de League of Legends a partir de datos del minuto 10 del juego. Se utilizan modelos de Machine Learning supervisados y no supervisados, así como un modelo de stacking para combinar los mejores resultados.

Se procesan datos crudos, se realiza análisis exploratorio, se entrenan y evalúan múltiples modelos, y finalmente se guarda un modelo final para despliegue.

Estructura de carpetas
1. data/

Contiene los datos utilizados en el proyecto, organizados en subcarpetas:

raw/: Datos en formato original, sin procesar.
Ejemplo: ranked_10min.csv

processed/: Datos procesados tras aplicar transformaciones, feature engineering y limpieza.
Ejemplo: processed.csv

train/: Datos de entrenamiento generados a partir de los datos procesados.
Ejemplo: train.csv

test/: Datos de prueba generados a partir de los datos procesados.
Ejemplo: test.csv

2. notebooks/

Contiene los notebooks Jupyter del desarrollo del proyecto:

01_Fuentes.ipynb: Adquisición de datos y unión de fuentes.

02_LimpiezaEDA.ipynb: Limpieza de datos, transformaciones, feature engineering y análisis exploratorio con visualizaciones.

03_Entrenamiento_Evaluacion.ipynb: Entrenamiento de modelos supervisados y no supervisados, hiperparametrización y evaluación de métricas.

3. src/

Archivos Python que implementan funcionalidades clave:

data_processing.py: Procesa los datos de data/raw/ y guarda los datasets en data/processed/.

training.py: Entrena los modelos a partir de los datos procesados y guarda los datasets de train/ y test/.

evaluation.py: Evalúa los modelos utilizando los datos de data/test/ y genera métricas de evaluación.

4. models/

Contiene los modelos entrenados y la configuración del modelo final:

trained_model_<nombre>.pkl – Modelos entrenados con identificadores únicos o nombres descriptivos.
Ejemplos: trained_model_logistic_regression.pkl, trained_model_random_forest.pkl, trained_model_xgboost.pkl, etc.

final_model.pkl – Modelo final seleccionado tras la evaluación.

stacking_config.yaml – Configuración del modelo final de stacking, incluyendo estimadores, hiperparámetros y umbrales de decisión.

Nota: Mantener un registro de métricas de cada modelo en docs/ o en un archivo de seguimiento es recomendable.

5. app_streamlit/

Contiene los archivos para desplegar el modelo final en una aplicación web:

app.py: Código de la aplicación Streamlit para predecir el ganador de un equipo.

requirements.txt: Dependencias necesarias para ejecutar la aplicación.

6. docs/

Documentación adicional del proyecto, incluyendo:

Memorias y reportes.

Presentaciones de resultados.

Seguimiento de métricas de modelos.

Pipeline del proyecto

Adquisición de datos: Se obtienen los datos crudos de partidas de League of Legends (minuto 10).

Procesamiento y limpieza: Se eliminan columnas irrelevantes, se renombra y unifica la nomenclatura, y se crean features derivados como diferencias entre equipos.

Exploración de datos: Se analizan correlaciones, distribuciones y relaciones mediante gráficos y heatmaps.

Entrenamiento de modelos: Se entrenan múltiples modelos supervisados (Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, XGBoost, KNN, SVC) y un modelo no supervisado (KMeans).

Hiperparametrización: Se optimizan hiperparámetros mediante GridSearchCV y RandomizedSearchCV.

Evaluación de modelos: Se calculan métricas de rendimiento como Accuracy, Precision, Recall, F1 y ROC-AUC.

Modelo final y stacking: Se crea un StackingClassifier combinando los mejores modelos y se guarda su configuración en YAML.

Despliegue: Se prepara la aplicación Streamlit para hacer predicciones en tiempo real según el estado de la partida.

Uso de la aplicación

Ejecutar la aplicación con Streamlit:

streamlit run app_streamlit/app.py


Seleccionar el equipo (azul o rojo) y obtener la probabilidad de victoria y un estado de alerta según umbrales definidos:

Probabilidad < 0.2: ⚠️ Alta probabilidad de perder

Probabilidad > 0.7: ✅ Probabilidad de ganar alta

Probabilidad intermedia: 🔹 Probabilidad intermedia

Dependencias principales

pandas, numpy

scikit-learn

xgboost

tensorflow / keras

matplotlib, seaborn

streamlit

Todas las dependencias necesarias para la aplicación están listadas en app_streamlit/requirements.txt.