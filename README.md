# Estimación de ingresos anuales de viviendas turísticas en Madrid

Modelo de regresión que estima los ingresos anuales de un alojamiento
turístico en Madrid a partir de sus características, su ubicación y el
perfil de su anfitrión.

Desplegado como API REST en un
[repositorio aparte](https://github.com/HimuraFresh/madrid-revenue-prediction-api),
con demo en [estimatupiso.onrender.com](https://estimatupiso.onrender.com).

---

## Por qué se rehizo

El proyecto nació como trabajo de equipo durante el bootcamp. Al revisarlo
individualmente aparecieron dos problemas de fondo: el dataset contenía un
solo trimestre en lugar de los tres previstos, y el modelo se entrenaba con
las mismas variables que Inside Airbnb usa para calcular el target.

Se reconstruyó entero: dataset regenerado, preprocesado calculado solo sobre
train y features limitadas a lo que un usuario puede aportar antes de
publicar un anuncio.

---

## Los datos

Tres snapshots de Madrid de 2025 publicados por
[Inside Airbnb](https://insideairbnb.com/get-the-data/): marzo, junio y
septiembre. Cada uno contiene el universo completo de anuncios activos en esa
fecha, así que un alojamiento que operó todo el año aparece en los tres.

El notebook `dataset_completo.ipynb` los unifica conservando la versión más
reciente de cada anuncio: 31.231 alojamientos con 78 variables.

Los datos brutos no se versionan. Para reconstruir el dataset hay que
descargar los tres ficheros, colocarlos en `data/raw/Airbnb_2025/` y ejecutar
ese notebook.

---

## El criterio que ordena el proyecto

El target `estimated_revenue_l365d` no es una cifra medida: Inside Airbnb lo
calcula multiplicando el precio por una ocupación que deriva del número de
reseñas. Ambos factores están en el dataset.

Pero el filtro que acabó decidiendo qué variables entran no fue ese, sino uno
de producto: **¿puede aportar este dato quien va a usar la API?**

Quien consulta está proyectando un piso que todavía no ha publicado. No tiene
ocupación del último año, ni reseñas, ni valoraciones, ni fechas de primera
reseña. Bajo ese criterio se retiran quince variables, incluidas las cinco con
mayor correlación con el objetivo.

Lo que queda describe el alojamiento y su anfitrión, que es exactamente la
información con la que cuenta un propietario antes de publicar.

---

## Flujo de trabajo

**Orden del notebook.** Se elimina el target nulo, se define X e y, se hace el
split y a partir de ahí todo el preprocesado se calcula sobre train y se
aplica a test. Los filtros de filas afectan solo a train; las correcciones de
valores inválidos, a ambos.

**Target.** Distribución muy sesgada a la derecha. Se le aplica `log1p` para
que el error que minimiza el entrenamiento sea relativo y no absoluto. Se
retiran los 4.645 anuncios con revenue exactamente cero, todos ellos sin
reseñas en el último año.

**Nulos.** Tratados según lo que significa la ausencia en cada caso: mediana,
moda, valores específicos, y categoría `Unknown` donde imputar la moda habría
situado el dato en el mejor tramo de una escala ordinal. Las tasas de respuesta
generan además una binaria, porque los anuncios sin ese dato facturan un 36%
menos.

**Outliers.** Corte por precio por plaza en lugar de precio absoluto, para no
eliminar alojamientos grandes de gama alta, que son oferta legítima. Se
detectaron anfitriones repitiendo el mismo precio en anuncios de capacidad
distinta, un patrón que inflaba el target.

**Feature engineering.** Ocho variables nuevas: antigüedad del anfitrión,
baño compartido, declaración de licencia, declaración de ubicación, número de
servicios, precio por plaza y escala ordinal de tiempo de respuesta.

**Preprocesado dentro del pipeline.** Escalado para las numéricas, one-hot para
las categóricas de baja cardinalidad y target encoding suavizado para los 127
barrios y los 21 distritos. Va dentro del pipeline para que se reajuste en cada
fold y el encoding nunca vea el target del conjunto de validación.

---

## Resultados

Cuatro candidatos bajo el mismo K-Fold de cinco particiones, RMSE sobre el
target en escala logarítmica.

| Modelo | RMSE CV |
|---|---|
| Baseline (mediana) | 1,327 |
| Ridge | 1,001 |
| Random Forest | 0,866 |
| XGBoost | 0,853 |
| XGBoost optimizado | 0,848 |

La optimización con `RandomizedSearchCV` mejora cinco milésimas, por debajo de
la desviación entre folds. El techo está en la información disponible, no en la
configuración del modelo.

**Evaluación final sobre test:**

| Métrica | Valor |
|---|---|
| RMSE (log) | 0,838 |
| R² (log) | 0,584 |
| MAE | 8.037 € |
| Error porcentual mediano | 43,3 % |

El RMSE en test queda por debajo del obtenido en validación cruzada, así que el
modelo generaliza sin señales de sobreajuste.

El uso razonable es comparativo: situar un alojamiento respecto a su zona y
estimar el efecto de cambiar precio o capacidad.

---

## Limitaciones

El target es una estimación de Inside Airbnb, no una cifra de ingresos real,
con los supuestos que eso arrastra: tasa fija de conversión de reseñas a
reservas y tope de ocupación del 70%.

Los datos son una fotografía de 2025 y no capturan la evolución de la
regulación ni de la demanda turística.

El modelo optimiza el error en escala logarítmica, y deshacer la transformación
con `expm1` devuelve aproximadamente la mediana condicional, no la media.

---

## Estructura del repositorio

madrid-rental-revenue-prediction/
├── data/
│ ├── raw/ datos de Inside Airbnb (no versionados)
│ └── processed/ dataset unificado y conjunto de test
├── models/
│ └── modelo_revenue_madrid.pkl
├── notebooks/
│ ├── dataset_completo.ipynb unificación de los tres snapshots
│ └── main.ipynb EDA, preprocesado y modelado
├── src/
│ └── utils/ funciones auxiliares
├── .gitignore
├── README.md
└── requirements.txt

---

## Stack

Python, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn y joblib.

---

## Autoría

Trabajo original desarrollado en equipo por Nazareth Montero, Javier Pascual,
Román Diaz y Sara Ruiz durante el bootcamp de Data Science e IA de The Bridge.

La revisión posterior, la corrección del dataset y el reentrenamiento del
modelo son trabajo individual de Román Diaz.
