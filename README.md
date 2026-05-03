# Irrigation Need Prediction - Kaggle Playground Series

Este repositorio contiene una solución para el desafío de **Kaggle Playground Series**, donde el objetivo es predecir la necesidad de riego de diferentes tipos de cultivos basándose en datos ambientales y de suelo[cite: 1].

## Descripción del Proyecto
El proyecto utiliza un dataset de **630,000 registros** para entrenamiento[cite: 1]. Se implementó un modelo de clasificación capaz de distinguir entre tres niveles de necesidad de riego: **Low**, **Medium** y **High**[cite: 1].

## Tecnologías y Librerías
- **Lenguaje:** Python
- **Librerías principales:**
  - `Pandas`: Manipulación y limpieza de datos[cite: 1].
  - `Scikit-learn`: Implementación del modelo y métricas de evaluación[cite: 1].

## Enfoque Técnico
1. **Preprocesamiento:**
   - Codificación de variables categóricas mediante `LabelEncoder` (Soil_Type, Crop_Type, etc.)[cite: 1].
   - Manejo de datos desbalanceados mediante el parámetro `class_weight='balanced'` en el modelo[cite: 1].
2. **Modelo:**
   - Se utilizó un **RandomForestClassifier** con 200 estimadores y una profundidad máxima de 20[cite: 1].
3. **Métrica de éxito:**
   - El modelo alcanzó un **Balanced Accuracy Score de 0.9611** en el conjunto de validación local[cite: 1].

## Estructura del Repositorio
- `src/main.py`: Script principal con el flujo de carga, entrenamiento y predicción[cite: 1].
- `data/`: Carpeta (excluida en git) que contiene los archivos `train.csv` y `test.csv`[cite: 1].
- `requirements.txt`: Lista de dependencias para replicar el entorno[cite: 1].

## Cómo ejecutar
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Colocar los archivos de datos de la competencia en la carpeta `/data`.
4. Ejecutar: `python src/main.py`.

---
*Proyecto desarrollado como parte de formación en Ingeniería en Sistemas de Información.*
