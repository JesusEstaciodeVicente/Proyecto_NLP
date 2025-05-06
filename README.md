# Clasificador de Noticias: Real vs. Falsa

## Descripción General

Proyecto profesional de clasificación binaria de titulares de noticias, distinguiendo entre noticias reales y falsas mediante técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) y aprendizaje automático. Incluye análisis de sentimiento y una interfaz profesional basada en Streamlit para visualización, explotación de resultados y logs en tiempo real.

---

## Estructura del Proyecto

```
Proyecto_NLP/
├── DATA/
│   ├── training_data.csv      # Datos de entrenamiento (etiquetados)
│   ├── testing_data.csv       # Datos de prueba (sin etiquetar)
│   └── predictions.csv        # Resultados de predicción
├── IMG/
│   └── Logo.jpg               # Logo de la aplicación
├── Proyecto_NLP.py            # Script principal (entrenamiento, predicción, interfaz)
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
└── ...                        # Otros archivos auxiliares
```

---

## Dependencias

Ver `requirements.txt` para la lista completa. Principales:
- Python >= 3.7
- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk, spacy, gensim
- streamlit, wordcloud, pillow
- joblib, tqdm, plotly, altair, openpyxl, xlrd

Instalación recomendada:
```bash
pip install -r requirements.txt
```

---

## Uso Rápido

### 1. Entrenamiento y Predicción
Ejecuta el script principal para entrenar modelos y generar predicciones:
```bash
python Proyecto_NLP.py
```
Esto:
- Preprocesa los datos
- Entrena y evalúa varios modelos (Logistic Regression, Naive Bayes, Random Forest, SVM, Gradient Boosting)
- Selecciona el mejor modelo
- Genera `DATA/predictions.csv` con las predicciones sobre el conjunto de prueba

### 2. Interfaz Web Profesional
Lanza la aplicación Streamlit para análisis interactivo y visualización avanzada:
```bash
streamlit run Proyecto_NLP.py app
```

- Incluye panel de navegación, visualización de métricas, análisis de sentimiento, gráficas y nubes de palabras.
- Todos los mensajes de estado y logs se muestran en la interfaz, no en la terminal.
- Botón profesional para mostrar resultados y visualizaciones avanzadas.

---

## Detalles Técnicos

- **Preprocesamiento:**
  - Limpieza, normalización, tokenización, lematización, eliminación de stopwords
  - Vectorización TF-IDF
- **Modelos:**
  - Logistic Regression, MultinomialNB, Random Forest, SVM, Gradient Boosting
  - Selección automática del mejor modelo por accuracy
  - Ensamblado opcional de los mejores modelos
- **Análisis de Sentimiento:**
  - VADER (NLTK)
- **Visualización:**
  - Streamlit UI profesional
  - Gráficos de barras, pastel, nubes de palabras, métricas y tablas interactivas
- **Persistencia:**
  - Modelos y resultados serializados (`best_model.pkl`, `model_results.pkl`)

---

## Ejemplo de Ejecución

1. Entrenamiento y predicción:
   ```bash
   python Proyecto_NLP.py
   ```
2. Interfaz web:
   ```bash
   streamlit run Proyecto_NLP.py app
   ```
3. Navega por las secciones: descripción, técnicas, ejecución, resultados, gráficas avanzadas.

---


