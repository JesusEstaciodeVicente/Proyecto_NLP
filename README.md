# 📰 Clasificador de Noticias Reales vs. Falsas

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Procesamiento%20de%20Texto-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Clasificación-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Visualización-red)

## 📋 Descripción del Proyecto

Este sistema implementa un clasificador de titulares de noticias utilizando técnicas avanzadas de procesamiento de lenguaje natural (NLP) para determinar si un titular corresponde a una noticia real o falsa. Además, proporciona un análisis de sentimiento para evaluar el tono emocional del texto.

![Demo](https://via.placeholder.com/800x400?text=Clasificador+de+Noticias)

## ✨ Características Principales

- **Clasificación Binaria**: Sistema de clasificación automática de titulares como reales (1) o falsos (0)
- **Múltiples Modelos**: Evaluación de diferentes algoritmos de aprendizaje automático (Regresión Logística, Naive Bayes, Random Forest)
- **Análisis de Sentimiento**: Evaluación del tono emocional del texto mediante VADER
- **Interfaz Interactiva**: Aplicación web desarrollada con Streamlit para facilitar la interacción con el modelo
- **Visualizaciones**: Representaciones gráficas de los resultados y análisis de sentimiento
- **Procesamiento de Texto**: Implementación de técnicas de limpieza, normalización y vectorización de texto

## 📊 Datos Utilizados

El proyecto trabaja con dos conjuntos de datos:

- **Datos de Entrenamiento** (`DATA/training_data.csv`): 
  - Contiene titulares de noticias etiquetados
  - **Etiqueta 0**: Noticia falsa
  - **Etiqueta 1**: Noticia real

- **Datos de Prueba** (`DATA/testing_data.csv`):
  - Contiene titulares sin etiquetar 
  - Todos los titulares tienen la etiqueta **2** como marcador
  - El objetivo es clasificarlos como falsos (0) o reales (1)
  - Las predicciones se guardan en `DATA/predictions.csv`

## 🛠️ Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal
- **pandas & numpy**: Manipulación y análisis de datos
- **scikit-learn**: Implementación de modelos de aprendizaje automático
- **NLTK**: Biblioteca para procesamiento de lenguaje natural
- **matplotlib & seaborn**: Visualización de datos
- **Streamlit**: Desarrollo de la interfaz de usuario web

## 🔧 Requisitos e Instalación

### Requisitos del Sistema

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Clonar el repositorio (opcional)
git clone https://github.com/usuario/clasificador-noticias.git
cd clasificador-noticias

# Instalar dependencias
pip install -r requirements.txt
```

## 📂 Estructura del Proyecto

```
clasificador-noticias/
│
├── Proyecto_NLP.py        # Script principal con implementación del modelo
├── app.py                 # Aplicación Streamlit mejorada
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Documentación
│
└── DATA/
    ├── training_data.csv  # Datos de entrenamiento (0 = falso, 1 = real)
    ├── testing_data.csv   # Datos de prueba (marcados con 2)
    └── predictions.csv    # Archivo generado con predicciones
```

## 🚀 Uso

### Entrenamiento del Modelo y Generación de Predicciones

```bash
python Proyecto_NLP.py
```

Este proceso realiza las siguientes operaciones:
1. Carga y preprocesamiento de los datos
2. Entrenamiento y evaluación de múltiples modelos
3. Selección del modelo con mejor rendimiento
4. Generación de predicciones para el conjunto de prueba
5. Almacenamiento del modelo entrenado para uso posterior

**Importante**: Los archivos CSV originales no se modifican durante este proceso. Las predicciones se guardan en un archivo separado.

### Ejecución de la Aplicación Web

```bash
# Versión mejorada (recomendada)
streamlit run app.py

# Versión básica alternativa
streamlit run Proyecto_NLP.py app
```

## 🖥️ Interfaz de Usuario

La aplicación Streamlit ofrece las siguientes funcionalidades:

- **Entrada de Texto**: Campo para ingresar titulares de noticias
- **Clasificación**: Predicción de la categoría (real o falsa) con nivel de confianza
- **Análisis de Sentimiento**: Evaluación del tono emocional con visualización gráfica
- **Preprocesamiento**: Visualización del texto procesado
- **Métricas**: Indicadores de confianza y composición del sentimiento

## 📝 Metodología

### 1. Preprocesamiento de Texto

- **Normalización**: Conversión a minúsculas
- **Limpieza**: Eliminación de caracteres especiales y números
- **Tokenización**: Segmentación del texto en unidades (tokens)
- **Eliminación de Stopwords**: Filtrado de palabras comunes sin valor semántico
- **Lematización**: Reducción de palabras a su forma base

### 2. Representación Vectorial

- **TF-IDF** (Term Frequency-Inverse Document Frequency): Transformación de texto en vectores numéricos ponderando la importancia de cada término

### 3. Modelos de Clasificación

- **Regresión Logística**: Modelo lineal para clasificación binaria
- **Naive Bayes**: Clasificador probabilístico basado en el teorema de Bayes
- **Random Forest**: Conjunto de árboles de decisión para mayor robustez

### 4. Análisis de Sentimiento

- **VADER** (Valence Aware Dictionary and sEntiment Reasoner): Herramienta especializada para análisis de sentimientos en textos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y commitea (`git commit -m 'Añade nueva funcionalidad'`)
4. Sube los cambios a tu fork (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está disponible bajo la licencia MIT. Consulta el archivo LICENSE para obtener más detalles.

## 📬 Contacto

Para preguntas o sugerencias, por favor contacta a: [correo@ejemplo.com]

---

Desarrollado como parte del proyecto NLP para Evolve Academy © 2023 