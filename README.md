# Estimación de ingresos anuales de viviendas turísticas en Madrid

Modelo de regresión que estima los ingresos anuales de un alojamiento turístico
en Madrid a partir de sus características, su actividad y su ubicación.

La variable objetivo es `estimated_revenue_l365d`: el ingreso estimado que ha
generado un anuncio en los 365 días anteriores a la fecha de recogida de datos.

El modelo está desplegado como API REST en un
[repositorio aparte](https://github.com/HimuraFresh/madrid-revenue-prediction-api),
con demo en [estimatupiso.onrender.com](https://estimatupiso.onrender.com).

---

## Estado actual

El proyecto se desarrolló originalmente como trabajo en equipo y está en
proceso de revisión individual. Dos cuestiones abiertas:

**Data leakage en el target.** Inside Airbnb no mide los ingresos, los calcula:
multiplica la ocupación estimada por el precio. Ambos factores están en el
propio dataset, y la ocupación se deriva a su vez del número de reseñas y de la
estancia mínima. El modelo inicial alcanzaba un R² de 0.9998, una cifra que por
sí sola ya indica que algo no encaja. Queda pendiente decidir qué variables
retirar y reentrenar aceptando la caída de métricas que eso implica.

**Dataset regenerado.** El fichero con el que se entrenó el modelo actual
contenía solo el snapshot de marzo de 2025, no la unión de los tres trimestres
que describía el notebook. Tras corregir la eliminación de duplicados, el
dataset pasa de 19.651 a 31.231 alojamientos.

Las métricas de este README corresponden al modelo antiguo y se actualizarán
tras el reentrenamiento.

---

## Datos

Los datos proceden de [Inside Airbnb](https://insideairbnb.com/get-the-data/),
que publica periódicamente una foto completa de los anuncios activos de cada
ciudad. Se utilizan tres snapshots de Madrid de 2025: marzo, junio y septiembre.

Cada snapshot contiene el universo completo de anuncios en esa fecha, de modo
que un alojamiento activo durante todo el año aparece en los tres. El notebook
`dataset_completo.ipynb` los unifica y conserva la versión más reciente de cada
anuncio, lo que deja 31.231 alojamientos con 78 variables.

Los datos brutos no están versionados. Para reconstruir el dataset hay que
descargar los tres ficheros de Inside Airbnb, colocarlos en
`data/raw/Airbnb_2025/` y ejecutar `dataset_completo.ipynb`.

Las variables cubren características del alojamiento, actividad y reseñas,
disponibilidad, restricciones de estancia y perfil del anfitrión.

---

## Flujo de trabajo

### Análisis exploratorio

El target presenta una distribución muy sesgada a la derecha, con una cola larga
de alojamientos de ingresos altos. Se le aplica `log1p` para reducir la
asimetría y limitar el peso de los valores extremos en el entrenamiento.

Las variables más correlacionadas con el target resultan ser
`estimated_occupancy_l365d`, `number_of_reviews`, `reviews_per_month` y
`number_of_reviews_ltm`. Esa correlación no es un hallazgo del análisis sino una
consecuencia de cómo se construye el target, y es el punto de partida del
problema de leakage descrito arriba.

### Preprocesado

Duplicados identificados por el `id` del anuncio. Valores nulos tratados según
el caso: eliminación de columnas muy incompletas, imputación por mediana o moda,
valores específicos y categoría `Unknown` donde la ausencia tiene significado.
Outliers acotados por winsorización sobre los percentiles 1 y 99.

### Feature engineering

One-hot encoding para los 21 distritos, mapeo binario y ordinal para las
categóricas de baja cardinalidad, y target encoding para barrio y tipo de
propiedad, que tienen demasiadas categorías para un one-hot.

El target encoding se calcula únicamente sobre el conjunto de entrenamiento y se
aplica a test mapeando esas medias, con la media global como valor de respaldo
para las categorías no vistas.

### Modelado

Se comparan Ridge como baseline lineal, Random Forest y XGBoost, todos con
K-Fold Cross Validation de 5 particiones, `shuffle=True` y `random_state=42`.

La métrica es RMSE sobre `revenue_log`. Para interpretar el error en euros se
deshace la transformación con `expm1`. Conviene tener presente que exponenciar
una predicción hecha en escala logarítmica devuelve aproximadamente la mediana
condicional, no la media.

XGBoost obtiene el mejor rendimiento y es el modelo que se optimiza con
`RandomizedSearchCV`, elegido sobre GridSearch por coste computacional.

La evaluación final se hace sobre un conjunto de test que no interviene en el
entrenamiento, y el modelo resultante se persiste con `joblib`.

---

## Limitaciones

Los datos son una fotografía del mercado en 2025 y no capturan la evolución
futura de la regulación, la demanda turística o la economía local.

El target es una estimación de Inside Airbnb, no una cifra de ingresos real, con
los supuestos que eso arrastra: una tasa de conversión de reseñas a reservas fija
y un tope de ocupación del 70%.

---

## Estructura del repositorio

```
madrid-rental-revenue-prediction/
├── data/
│   ├── raw/                  datos de Inside Airbnb (no versionados)
│   └── processed/            dataset unificado y conjuntos de test
├── models/
│   └── modelo_optimizado.pkl
├── notebooks/
│   ├── dataset_completo.ipynb   unificación de los tres snapshots
│   └── main.ipynb               EDA, preprocesado, modelado
├── src/
│   └── utils/                funciones auxiliares
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Tecnologías

Python, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn y joblib.

---

## Autoría

Trabajo original desarrollado en equipo por Nazareth Montero, Javier Pascual,
Román Diaz y Sara Ruiz durante el bootcamp de Data Science e IA de The Bridge.

La revisión posterior, la corrección del dataset y el reentrenamiento del modelo
son trabajo individual de Román Diaz.