import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
import os
import sys
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
import pickle
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import time
import joblib
from tqdm import tqdm
import warnings
import base64
from PIL import Image
from io import BytesIO

# Filtrar advertencias (warnings) para que la salida sea más limpia
# Esto evitará que se muestren las advertencias de convergencia y otros warnings comunes
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)

# Definir semilla aleatoria global para reproducibilidad
RANDOM_SEED = 1976
BOTONAZO = 0

############################################
####### MOSTRAR MENSAJES DE PROGRESO #########
############################################
def print_status(message, verbose=True):
    """Muestra un mensaje de estado formateado con la hora actual"""
    if verbose:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{current_time}] {message}")
        sys.stdout.flush()  # Forzar que el mensaje se muestre inmediatamente

# Iniciar con mensajes de estado
print_status("Iniciando procesamiento de lenguaje natural...")
print_status("Configurando entorno y dependencias...")

# Descargar recursos de NLTK con mensajes de estado
print_status("Descargando recursos de NLTK (si no están ya instalados)...")
for resource in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']:
    print_status(f"  - Descargando recurso NLTK: {resource}")
    nltk.download(resource, quiet=True)

print_status("Recursos NLTK descargados correctamente.")

############################################
####### PREPROCESAR TEXTO #########
############################################
def preprocess_text(text):
    if isinstance(text, str):
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Eliminar stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lematización
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Unir tokens en un texto limpio
        return ' '.join(tokens)
    else:
        return ''

############################################
####### ANALIZAR SENTIMIENTO #########
############################################
def analyze_sentiment(text):
    if isinstance(text, str):
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        return sentiment
    else:
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

############################################
####### NORMALIZAR ARCHIVO CSV #########
############################################
def normalize_csv_file(file_path):
    """Normaliza un archivo CSV si es necesario, asegurando formato consistente"""
    print_status(f"Verificando y normalizando archivo: {file_path}")
    
    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        print_status(f"ERROR: El archivo {file_path} no existe")
        return False
    
    # Verificar tamaño del archivo
    file_size = os.path.getsize(file_path) / 1024  # tamaño en KB
    print_status(f"  - Tamaño del archivo: {file_size:.2f} KB")
    
    # Verificar si el archivo necesita normalización
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_lines = [file.readline().strip() for _ in range(5)]
            
        # Contar líneas en total
        with open(file_path, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file)
        
        print_status(f"  - Total de líneas: {total_lines}")
        print_status(f"  - Primeras líneas de ejemplo:")
        for i, line in enumerate(first_lines, 1):
            print_status(f"    {i}: {line}")
        
        return True
    except Exception as e:
        print_status(f"ERROR al analizar archivo: {str(e)}")
        return False

############################################
####### CARGAR Y PREPROCESAR DATOS #########
############################################
def load_data(print_status=print_status):
    print_status("Iniciando carga de datos...")
    
    # Verificar y normalizar archivos si es necesario
    training_file = 'DATA/training_data.csv'
    testing_file = 'DATA/testing_data.csv'
    
    print_status("Verificando archivos de datos...")
    normalize_csv_file(training_file)
    normalize_csv_file(testing_file)
    
    # Cargar datos usando un enfoque manual para separar etiqueta y texto
    print_status("Cargando archivos CSV...")
    
    # Función para procesar manualmente los archivos
    def process_file(file_path, print_status):
        labels = []
        titles = []
        line_count = 0
        
        print_status(f"Procesando archivo: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8-sig') as file:  # Cambiado a utf-8-sig para manejar BOM automáticamente
            for line_number, line in enumerate(file, 1):
                line_count += 1
                # Mostrar progreso cada 1000 líneas
                if line_count % 1000 == 0:
                    print_status(f"  - Procesadas {line_count} líneas...")
                
                # Limpiar la línea y eliminar caracteres no deseados
                line = line.strip()
                
                # Comprobar si la línea contiene un tabulador como separador
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    # Dividir por el primer espacio o grupo de espacios como alternativa
                    parts = line.split(None, 1)
                
                if len(parts) == 2:
                    label, title = parts
                    # Eliminar cualquier carácter BOM u otros caracteres especiales que puedan quedar
                    label = label.replace('\ufeff', '').strip()
                    # Limpiar y normalizar la etiqueta
                    try:
                        # Intentar convertir directamente a entero como validación
                        int(label)
                        labels.append(label)
                        titles.append(title)
                    except ValueError:
                        print_status(f"  - ADVERTENCIA: Etiqueta no numérica en línea {line_number}: '{label}'")
                else:
                    print_status(f"  - ADVERTENCIA: Línea {line_number} ignorada (formato incorrecto): {line}")
        
        print_status(f"  - Completado. Procesadas {line_count} líneas, obtenidos {len(labels)} registros válidos.")
        
        # Crear DataFrame
        df = pd.DataFrame({
            'label': labels,
            'title': titles
        })
        
        return df
    
    # Procesar archivos
    print_status("Procesando archivo de entrenamiento...")
    train_data = process_file(training_file, print_status)
    
    print_status("Procesando archivo de prueba...")
    test_data = process_file(testing_file, print_status)
    
    # Verificar la estructura de los datos
    print_status(f"Estructura de datos de entrenamiento: {train_data.shape}")
    print_status(f"Estructura de datos de prueba: {test_data.shape}")
    
    print_status("Primeras 3 filas de train_data:")
    print_status(str(train_data.head(3)))
    print_status("Primeras 3 filas de test_data:")
    print_status(str(test_data.head(3)))
    
    # Limpiar caracteres BOM (Byte Order Mark) en las etiquetas antes de convertir a enteros
    print_status("Limpiando caracteres especiales en las etiquetas...")
    train_data['label'] = train_data['label'].str.replace('\ufeff', '').str.strip()
    test_data['label'] = test_data['label'].str.replace('\ufeff', '').str.strip()
    
    # Mapear etiquetas si es necesario (por ejemplo si son strings o si hay etiquetas diferentes como '2')
    print_status("Verificando y normalizando etiquetas...")
    # Verificar si hay etiquetas diferentes de 0 y 1 en los datos de prueba
    unique_test_labels = test_data['label'].unique()
    print_status(f"Etiquetas únicas en datos de prueba: {unique_test_labels}")
    
    # En caso de que haya etiquetas como '2', mapearlas a valores válidos (0,1)
    # Asumimos que 0=falso, 1=real y cualquier otro valor como 2 se trata como 1 (real)
    if '2' in unique_test_labels:
        print_status("Encontrada etiqueta '2' en datos de prueba. Mapeando a 1 (real)...")
        test_data['label'] = test_data['label'].replace('2', '1')
    
    # Asegurar que las etiquetas sean numéricas
    print_status("Convirtiendo etiquetas a formato numérico...")
    train_data['label'] = train_data['label'].astype(int)
    test_data['label'] = test_data['label'].astype(int)
    
    print_status(f"Tipos de datos en train_data['label']: {train_data['label'].dtype}")
    print_status(f"Valores únicos en train_data['label']: {train_data['label'].unique()}")
    print_status(f"Valores únicos en test_data['label']: {test_data['label'].unique()}")
    
    # Preprocesar los títulos
    print_status("Preprocesando datos de entrenamiento...")
    print_status("  - Aplicando limpieza y normalización de texto...")
    train_data['title_processed'] = train_data['title'].apply(preprocess_text)
    
    print_status("Analizando sentimiento de los datos de entrenamiento...")
    train_data['sentiment'] = train_data['title'].apply(lambda x: analyze_sentiment(x)['compound'])
    
    # Extraer características adicionales de los titulares
    print_status("Extrayendo características adicionales...")
    train_data['title_length'] = train_data['title'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    train_data['word_count'] = train_data['title'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    train_data['avg_word_length'] = train_data['title'].apply(lambda x: 
                                                 np.mean([len(word) for word in x.split()]) 
                                                 if isinstance(x, str) and len(x.split()) > 0 else 0)
    
    print_status("Carga y preprocesamiento de datos completados.")
    return train_data, test_data

############################################
####### ENTRENAR Y EVALUAR MODELOS #########
############################################
def train_evaluate_models(train_data, print_status=print_status):
    print_status("Preparando datos para entrenamiento y validación...")
    # Dividir datos para entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        train_data['title_processed'], 
        train_data['label'], 
        test_size=0.2, 
        random_state=RANDOM_SEED,
        stratify=train_data['label']  # Asegurar distribución similar en ambos conjuntos
    )
    
    print_status(f"Conjunto de entrenamiento: {len(X_train)} muestras")
    print_status(f"Conjunto de validación: {len(X_val)} muestras")
    
    # Crear una instancia de validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    print_status(f"Configurada validación cruzada: {cv.get_n_splits()} particiones")
    
    # Configuraciones para TF-IDF con menos combinaciones para eficiencia
    print_status("Configurando parámetros para vectorización TF-IDF...")
    tfidf_params = {
        'tfidf__max_features': [5000],
        'tfidf__min_df': [2],
        'tfidf__max_df': [0.9],
        'tfidf__ngram_range': [(1, 2)],
        'tfidf__use_idf': [True],
        'tfidf__norm': ['l2']
    }
    
    # Definir modelos a probar con grids eficientes
    print_status("Configurando modelos a evaluar...")
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {
                'classifier__C': [1.0],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear'],
                'classifier__max_iter': [1000],
                'classifier__random_state': [RANDOM_SEED]
            }
        },
        'Naive Bayes': {
            'model': MultinomialNB(),
            'params': {
                'classifier__alpha': [1.0]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'classifier__n_estimators': [100],
                'classifier__random_state': [RANDOM_SEED]
            }
        },
        'SVM': {
            'model': LinearSVC(),
            'params': {
                'classifier__C': [1.0],
                'classifier__max_iter': [1000],
                'classifier__random_state': [RANDOM_SEED]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'classifier__n_estimators': [100],
                'classifier__learning_rate': [0.1],
                'classifier__random_state': [RANDOM_SEED]
            }
        }
    }
    
    print_status(f"Se evaluarán {len(models)} modelos diferentes")
    
    best_model = None
    best_accuracy = 0
    results = {}
    
    # Entrenar y evaluar cada modelo con búsqueda de hiperparámetros
    for model_index, (name, model_info) in enumerate(models.items(), 1):
        print_status(f"\n[{model_index}/{len(models)}] Entrenando {name} con búsqueda de hiperparámetros...")
        start_time = time.time()
        
        # Crear pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', model_info['model'])
        ])
        
        # Combinar parámetros
        param_grid = {**tfidf_params}
        for param, values in model_info['params'].items():
            param_grid[param] = values
        
        print_status(f"Configuración de búsqueda para {name}: {len(param_grid)} parámetros a optimizar")
        
        # Crear búsqueda de hiperparámetros
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Entrenar modelo con búsqueda de hiperparámetros
        print_status(f"Iniciando búsqueda de hiperparámetros para {name}...")
        grid_search.fit(X_train, y_train)
        
        # Obtener el mejor modelo
        best_pipeline = grid_search.best_estimator_
        
        # Evaluar en conjunto de validación
        print_status(f"Evaluando mejor modelo de {name} en conjunto de validación...")
        y_pred = best_pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        # Intentar obtener predicciones de probabilidad para ROC AUC si el modelo lo soporta
        try:
            y_proba = best_pipeline.predict_proba(X_val)
            roc_auc = roc_auc_score(y_val, y_proba[:, 1])
        except:
            roc_auc = None
        
        # Generar informe de clasificación
        report = classification_report(y_val, y_pred)
        
        # Calcular tiempo de entrenamiento
        training_time = time.time() - start_time
        
        print_status(f"{name} - Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print_status(f"  - {param}: {value}")
        
        print_status(f"{name} - Métricas de rendimiento:")
        print_status(f"  - Accuracy: {accuracy:.4f}")
        print_status(f"  - Precision: {precision:.4f}")
        print_status(f"  - Recall: {recall:.4f}")
        print_status(f"  - F1: {f1:.4f}")
        if roc_auc:
            print_status(f"  - ROC AUC: {roc_auc:.4f}")
        print_status(f"  - Tiempo de entrenamiento: {training_time:.2f} segundos")
        
        print(report)
        
        # Guardar resultados
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'report': report,
            'model': best_pipeline,
            'best_params': grid_search.best_params_,
            'training_time': training_time
        }
        
        # Actualizar mejor modelo
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_pipeline
            print_status(f"Nuevo mejor modelo encontrado: {name} con accuracy {accuracy:.4f}")
    
    # Entrenar un ensamble de los mejores modelos
    print_status("\nEntrenando modelo de ensamble con los mejores modelos...")
    
    # Seleccionar los 3 mejores modelos por precisión
    top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
    print_status(f"Top 3 modelos seleccionados para ensamble:")
    for i, (name, _) in enumerate(top_models, 1):
        print_status(f"  {i}. {name} - Accuracy: {results[name]['accuracy']:.4f}")
    
    # Crear un ensamble con los mejores modelos
    # En vez de usar los pipelines completos, usamos solo los clasificadores base
    estimators = []
    for name, result in results.items():
        if name in [model[0] for model in top_models]:
            # Extraer el clasificador base del pipeline
            clf = result['model'].named_steps['classifier']
            estimators.append((name, clf))
    
    if len(estimators) > 1:
        # Crear pipeline de ensamble
        print_status("Configurando pipeline de ensamble...")
        ensemble_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.9,
                ngram_range=(1, 2),
                use_idf=True
            )),
            ('ensemble', VotingClassifier(estimators=estimators, voting='hard'))
        ])
        
        # Entrenar ensamble
        print_status("Entrenando modelo de ensamble...")
        ensemble_start_time = time.time()
        ensemble_pipeline.fit(X_train, y_train)
        ensemble_training_time = time.time() - ensemble_start_time
        
        # Evaluar ensamble
        print_status("Evaluando modelo de ensamble...")
        y_pred = ensemble_pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        
        print_status(f"Ensemble Model - Accuracy: {accuracy:.4f}")
        print_status(f"Ensemble Model - Tiempo de entrenamiento: {ensemble_training_time:.2f} segundos")
        print(report)
        
        # Guardar resultados del ensamble
        results['Ensemble'] = {
            'accuracy': accuracy,
            'report': report,
            'model': ensemble_pipeline,
            'training_time': ensemble_training_time
        }
        
        # Actualizar mejor modelo si el ensamble es mejor
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = ensemble_pipeline
            print_status(f"El modelo de ensamble es ahora el mejor modelo con accuracy {accuracy:.4f}")
    
    # Guardar el mejor modelo
    print_status(f"Guardando el mejor modelo (accuracy: {best_accuracy:.4f})...")
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Guardar todos los resultados para análisis posterior
    print_status("Guardando resultados de todos los modelos para análisis...")
    joblib.dump(results, 'model_results.pkl')
    
    # Mostrar comparativa de modelos
    print_status("\nComparativa final de todos los modelos:")
    print("="*80)
    print(f"{'Modelo':<20} | {'Accuracy':<10} | {'F1':<10} | {'Tiempo (s)':<10}")
    print("-"*80)
    for name, result in results.items():
        acc = result.get('accuracy', 0)
        f1 = result.get('f1', 0)
        time_taken = result.get('training_time', 0)
        print(f"{name:<20} | {acc:<10.4f} | {f1:<10.4f} | {time_taken:<10.2f}")
    print("="*80)
    
    return results, best_model

############################################
####### HACER PREDICCIONES #########
############################################
def make_predictions(test_data, model, print_status=print_status):
    print_status("Preprocesando datos de prueba...")
    test_data_copy = test_data.copy()  # Trabajar con una copia para no modificar el original
    test_data_copy['title_processed'] = test_data_copy['title'].apply(preprocess_text)
    
    print_status("Haciendo predicciones...")
    test_data_copy['predicted_label'] = model.predict(test_data_copy['title_processed'])
    
    # Crear un nuevo DataFrame para las predicciones
    predictions = pd.DataFrame()
    predictions['label'] = test_data_copy['predicted_label']
    predictions['title'] = test_data_copy['title']
    
    # Guardar predicciones en un nuevo archivo en el formato correcto
    output_path = 'DATA/predictions.csv'
    
    print_status(f"Guardando predicciones en {output_path}...")
    
    # Guardar en el mismo formato que los archivos originales
    with open(output_path, 'w', encoding='utf-8') as f:
        for index, row in tqdm(predictions.iterrows(), total=len(predictions), desc="Guardando predicciones"):
            f.write(f"{row['label']}       {row['title']}\n")
    
    print_status(f"Predicciones guardadas en: {output_path}")
    print_status(f"Total de noticias clasificadas como reales (1): {predictions['label'].sum()}")
    print_status(f"Total de noticias clasificadas como falsas (0): {len(predictions) - predictions['label'].sum()}")
    
    return predictions

############################################
####### FUNCIÓN PRINCIPAL #########
############################################
def main():
    print_status("="*80)
    print_status("INICIANDO PROYECTO DE CLASIFICACIÓN DE NOTICIAS CON NLP")
    print_status("="*80)
    
    # Verificar archivos y directorios
    print_status("Verificando estructura de directorios y archivos...")
    if not os.path.exists('DATA'):
        print_status("ERROR: No se encontró el directorio DATA")
        return
    
    # Cargar datos
    print_status("Iniciando fase de carga y preprocesamiento de datos...")
    train_data, test_data = load_data()
    
    print_status(f"Datos de entrenamiento cargados: {len(train_data)} titulares")
    print_status(f"Datos de prueba cargados: {len(test_data)} titulares")
    
    # Distribución de etiquetas en datos de entrenamiento
    print_status(f"Distribución de etiquetas en entrenamiento:")
    
    real_count = train_data['label'].sum()
    false_count = len(train_data) - real_count
    total_count = len(train_data)
    
    print_status(f"- Noticias reales (1): {real_count} ({real_count/total_count*100:.2f}%)")
    print_status(f"- Noticias falsas (0): {false_count} ({false_count/total_count*100:.2f}%)")
    
    # Entrenar y evaluar modelos
    print_status("\nIniciando fase de entrenamiento de modelos...")
    results, best_model = train_evaluate_models(train_data)
    
    # Encontrar el modelo con mejor rendimiento
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    print_status(f"\nMejor modelo: {best_model_name} con precisión de {best_accuracy:.4f}")
    
    # Hacer predicciones en conjunto de prueba
    print_status("\nIniciando fase de generación de predicciones para el conjunto de prueba...")
    predictions = make_predictions(test_data, best_model)
    
    print_status("\nProceso completo. Archivo de predicciones generado.")
    print_status("="*80)

############################################
####### INTERFAZ STREAMLIT #########
############################################
def get_image_base64(img_path):
    img = Image.open(img_path).convert("RGBA").resize((120, 120))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def run_app():
    st.set_page_config(
        page_title="Clasificador de Noticias Reales vs. Falsas",
        page_icon="📰",
        layout="wide"
    )
    st.markdown("""
    <style>
    /* Color principal azul para botones y acentos */
    .stButton>button {
        background-color: #1976D2 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 18px !important;
        font-weight: bold;
        padding: 0.5em 2em;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.15);
    }
    .stButton>button:hover {
        background-color: #1565C0 !important;
        color: #fff !important;
    }
    .stSidebar, .css-6qob1r, .css-1d391kg, .css-1lcbmhc, .css-1v0mbdj {
        background-color: #E3F2FD !important;
    }
    .st-bb, .st-c6, .st-cg, .st-cj, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
        color: #1976D2 !important;
    }
    .stApp {
        background-color: #F5FAFF !important;
    }
    .big-font {font-size:22px !important; color: #1976D2 !important;}
    .metric-label {font-size:16px !important; color: #1976D2 !important;}
    h1, h2, h3, h4, h5, h6 {
        color: #1976D2 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1976D2 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title('📰 Clasificador de Noticias Reales vs. Falsas')

    # Mostrar logo redondo y centrado en el sidebar
    try:
        img_base64 = get_image_base64(r"D:/YUGULO/Proyectos/Compartidos/Proyecto_NLP/IMG/Logo.jpg")
        st.sidebar.markdown(
            f'''
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_base64}" style="border-radius: 50%; width: 120px; height: 120px; object-fit: cover; border: 3px solid #ddd;" />
            </div>
            ''',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.write("Logo no disponible")

    st.sidebar.title("Panel de Opciones")
    seccion = st.sidebar.radio("Ir a:", [
        "Descripción y Archivos",
        "Técnicas y Flujo",
        "Ejecución y Resultados"
    ])
    st.sidebar.markdown("---")
    st.sidebar.info("Proyecto de Jesús Estacio de Vicente | Desarrollado para Evolve Academy | 2025")

    if seccion == "Descripción y Archivos":
        st.header("¿De qué trata el proyecto?")
        st.markdown("""
        Este sistema implementa un clasificador de titulares de noticias utilizando técnicas avanzadas de procesamiento de lenguaje natural (NLP) para determinar si un titular corresponde a una noticia real o falsa. Además, proporciona un análisis de sentimiento para evaluar el tono emocional del texto.
        
        **Archivos principales:**
        - `DATA/training_data.csv`: Contiene titulares de noticias etiquetados (0 = falso, 1 = real).
        - `DATA/testing_data.csv`: Contiene titulares sin etiquetar (etiqueta 2 como marcador).
        - `DATA/predictions.csv`: Archivo generado con las predicciones del modelo (0 = falso, 1 = real).
        
        El objetivo es entrenar un modelo con los datos de entrenamiento, predecir los titulares del archivo de prueba y guardar los resultados en el archivo de predicciones.
        """)
    elif seccion == "Técnicas y Flujo":
        st.header("Técnicas aplicadas y orden del proceso")
        st.markdown("""
        El flujo del proyecto es el siguiente:
        1. **Carga y preprocesamiento de datos**: Limpieza, normalización, tokenización, eliminación de stopwords y lematización de los titulares.
        2. **Extracción de características**: Vectorización de texto usando TF-IDF y extracción de métricas adicionales (longitud, número de palabras, etc.).
        3. **Entrenamiento de modelos**: Se prueban varios algoritmos de clasificación (Regresión Logística, Naive Bayes, Random Forest, SVM, Gradient Boosting) y se selecciona el mejor según su precisión.
        4. **Evaluación y comparación**: Se comparan los modelos usando métricas como accuracy, F1, tiempo de entrenamiento y se visualizan los resultados.
        5. **Predicción sobre datos de prueba**: El mejor modelo predice las etiquetas de los titulares de prueba y se guardan en `DATA/predictions.csv`.
        6. **Análisis de sentimiento**: Se utiliza VADER para analizar el tono emocional de los titulares.
        
        Todo el proceso está automatizado y optimizado para obtener el mejor rendimiento posible.
        """)
        st.info("Se podrá desplazar hacia abajo para ver todo el flujo si fuera necesario.")
    elif seccion == "Ejecución y Resultados":
        st.header("Ejecución del proceso y resultados")
        st.markdown("""
        Al pulsar el botón **COMENZAR**, se iniciará todo el proceso: carga de datos, entrenamiento, evaluación, predicción y generación de resultados. Los mensajes de avance y resultados aparecerán aquí en pantalla.
        """)
        status_msgs = []
        status_placeholder = st.empty()
        copy_placeholder = st.empty()
        if 'procesando' not in st.session_state:
            st.session_state['procesando'] = False
        if 'BOTONAZO' not in st.session_state:
            st.session_state['BOTONAZO'] = 0
        boton = st.button("COMENZAR", key="comenzar_btn", disabled=st.session_state['procesando'])
        def print_status(message, verbose=True):
            if verbose:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                status_msgs.append(f"[{current_time}] {message}")
                status_placeholder.markdown(
                    f'<div id="mensajes-box" style="height:350px;overflow-x:auto;overflow-y:auto;background:#f9f9f9;border:1px solid #ddd;padding:10px;font-family:monospace;font-size:15px;white-space:pre;">' +
                    '<br>'.join(status_msgs) + '</div>',
                    unsafe_allow_html=True
                )
        if boton:
            st.session_state['procesando'] = True
            with st.spinner('Trabajando. Un momento, por favor...'):
                try:
                    print_status("Iniciando procesamiento de lenguaje natural...")
                    print_status("Configurando entorno y dependencias...")
                    print_status("Descargando recursos de NLTK (si no están ya instalados)...")
                    import nltk
                    for resource in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']:
                        print_status(f"  - Descargando recurso NLTK: {resource}")
                        nltk.download(resource, quiet=True)
                    print_status("Recursos NLTK descargados correctamente.")
                    import sys
                    print_status(f"Ejecutando en Python {sys.version}")
                    print_status(f"Semilla aleatoria: {RANDOM_SEED}")
                    print_status("Iniciando aplicación Streamlit...")
                    train_data, test_data = load_data(print_status=print_status)
                    results, best_model = train_evaluate_models(train_data, print_status=print_status)
                    predictions = make_predictions(test_data, best_model, print_status=print_status)
                    print_status("\nProceso completo. Archivo de predicciones generado.")
                    st.success("¡Proceso finalizado! Revisa el archivo DATA/predictions.csv y los resultados generados.")
                    st.session_state['BOTONAZO'] = 1
                except Exception as e:
                    st.error(f"Ocurrió un error durante la ejecución: {e}")
            st.session_state['procesando'] = False
        # BOTONAZO: mostrar botón grande, verde, centrado y funcional para ver resultados
        if st.session_state['BOTONAZO'] == 1:
            # Botón grande, verde, centrado y funcional para ver resultados
            st.markdown("""
                <style>
                div.stButton > button#ver_resultados_btn {
                    background: #43a047;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    border: none;
                    border-radius: 14px;
                    padding: 20px 60px;
                    cursor: pointer;
                    box-shadow: 0 2px 8px rgba(67,160,71,0.15);
                    letter-spacing: 2px;
                    margin: 30px auto 20px auto;
                    display: block;
                }
                </style>
            """, unsafe_allow_html=True)
            col_pred = st.columns([1,2,1])
            with col_pred[1]:
                mostrar = st.button("VER RESULTADOS", key="ver_resultados_btn")
            if mostrar:
                import pandas as pd
                try:
                    df_pred = pd.read_csv("DATA/predictions.csv", sep="       ", names=["label", "title"], engine="python")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h3 style='color:#43a047;text-align:center;'>VERDADERO</h3>", unsafe_allow_html=True)
                        st.dataframe(df_pred[df_pred['label']==1][['title']].rename(columns={'title':'Titular'}), use_container_width=True)
                    with col2:
                        st.markdown("<h3 style='color:#d32f2f;text-align:center;'>FALSO</h3>", unsafe_allow_html=True)
                        st.dataframe(df_pred[df_pred['label']==0][['title']].rename(columns={'title':'Titular'}), use_container_width=True)
                except Exception as e:
                    st.error(f"No se pudo cargar el archivo de predicciones: {e}")

# El bloque if __name__ == '__main__' solo ejecuta main() o run_app(), sin imprimir mensajes directamente.
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'app':
        run_app()
    else:
        main()
