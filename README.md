# 📊 Proyecto de Análisis de TED Talks - Arquitectura Modular

## 🎯 Descripción

Este proyecto implementa un análisis completo de TED Talks utilizando técnicas de NLP y Machine Learning, organizado en módulos Python reutilizables para máxima flexibilidad y mantenibilidad.

## 📁 Estructura del Proyecto

```
data-extraction-TED-Talks-transcripts-for-NLP/
├── ted_talks_en.csv                 # Dataset principal
├── versionmanuel.ipynb             # Notebook con ejemplos de uso
├── modules/                        # Módulos Python organizados
│   ├── __init__.py                # Módulo principal e importaciones
│   ├── environment_setup.py       # Configuración del ambiente
│   ├── data_cleaner.py            # Limpieza profesional de datos
│   ├── nlp_processor.py           # Procesamiento de lenguaje natural
│   ├── visualizer.py              # Visualizaciones y gráficos
│   └── ml_models.py               # Modelos de machine learning
└── README.md                      # Este archivo
```

## 🚀 Inicio Rápido

### Opción 1: Análisis Completo Automatizado
```python
from modules import quick_start

# Ejecuta todo el pipeline automáticamente
analyzer, results = quick_start('ted_talks_en.csv')
```

### Opción 2: Control Paso a Paso
```python
from modules import TedTalkAnalyzer

analyzer = TedTalkAnalyzer()
analyzer.setup_environment()        # Configurar ambiente
analyzer.load_data('ted_talks_en.csv')  # Cargar datos
analyzer.clean_data()               # Limpiar datos
analyzer.process_nlp_features()     # Procesar NLP
analyzer.create_visualizations()    # Crear gráficos
analyzer.train_models()             # Entrenar modelos ML
```

### Opción 3: Funciones Específicas
```python
# Importar solo lo que necesitas
from modules.data_cleaner import clean_dataset_professional
from modules.nlp_processor import analyze_sentiment
from modules.visualizer import create_wordcloud
from modules.ml_models import TedTalkClassifier

# Usar funciones individualmente
df_clean, log = clean_dataset_professional(df)
sentiment = analyze_sentiment("Amazing talk!")
create_wordcloud(text_data, "Mi Nube de Palabras")
```

## 📚 Módulos Disponibles

### 🔧 `environment_setup.py`
Configura el ambiente y descarga dependencias necesarias.

**Funciones principales:**
- `setup_environment()` - Instala todas las dependencias
- `download_transformer_models()` - Descarga modelos de Hugging Face
- `check_device()` - Verifica disponibilidad de GPU
- `get_environment_info()` - Información del ambiente configurado

### 🧹 `data_cleaner.py`
Limpieza profesional de datos siguiendo estándares de la industria.

**Funciones principales:**
- `clean_dataset_professional(df)` - Pipeline completo de limpieza
- `remove_outliers_iqr(df, column)` - Elimina outliers usando IQR
- `create_popularity_categories(df)` - Crea categorías de popularidad
- `calculate_data_quality(df)` - Calcula puntuación de calidad
- `validate_data_quality(df)` - Valida calidad de datos

### 🔤 `nlp_processor.py`
Procesamiento avanzado de lenguaje natural.

**Funciones principales:**
- `process_text_features(df)` - Procesa todas las características de texto
- `analyze_sentiment(text)` - Análisis de sentimientos con TextBlob
- `extract_named_entities(text, nlp_model)` - Extrae entidades con spaCy
- `extract_text_features(text)` - Características básicas de texto
- `create_word_frequency_analysis(df)` - Análisis de frecuencia de palabras

### 📊 `visualizer.py`
Visualizaciones profesionales y gráficos interactivos.

**Funciones principales:**
- `create_data_overview_plots(df)` - Gráficos de resumen del dataset
- `create_correlation_heatmap(df)` - Matriz de correlación
- `create_sentiment_analysis_plots(df)` - Gráficos de sentimientos
- `create_text_features_plots(df)` - Gráficos de características textuales
- `create_wordcloud(text_data)` - Nube de palabras
- `create_interactive_plots(df)` - Gráficos interactivos con Plotly

### 🤖 `ml_models.py`
Modelos de machine learning y evaluación.

**Clases y funciones principales:**
- `TedTalkClassifier` - Clase principal para clasificación
- `create_ml_pipeline(df)` - Pipeline completo de ML
- Modelos incluidos: Random Forest, Gradient Boosting, Logistic Regression, SVM
- Evaluación automática con métricas: Accuracy, Precision, Recall, F1-Score, AUC

### 🎯 `__init__.py`
Módulo principal que orquesta todo el análisis.

**Clase principal:**
- `TedTalkAnalyzer` - Orquesta todo el pipeline de análisis
- `quick_start()` - Función de inicio rápido

## 🔥 Características Principales

### ✨ Funcionalidades de NLP
- **Análisis de Sentimientos**: Polaridad y subjetividad con TextBlob
- **Entidades Nombradas**: Extracción con spaCy (personas, organizaciones, ubicaciones)
- **Características Textuales**: Longitud, diversidad léxica, complejidad
- **TF-IDF**: Vectorización para modelos de ML
- **Frecuencia de Palabras**: Análisis de términos más comunes

### 📈 Modelos de Machine Learning
- **Random Forest**: Con optimización de hiperparámetros
- **Gradient Boosting**: Para mejores predicciones
- **Logistic Regression**: Modelo base interpretable
- **SVM**: Para clasificación no lineal
- **Evaluación Completa**: Validación cruzada, matrices de confusión, métricas

### 🎨 Visualizaciones
- **Gráficos Estáticos**: matplotlib y seaborn
- **Gráficos Interactivos**: Plotly para exploración
- **Nubes de Palabras**: WordCloud personalizable
- **Matrices de Correlación**: Análisis de relaciones
- **Distribuciones**: Histogramas y box plots

### 🛡️ Calidad de Datos
- **Limpieza Profesional**: Eliminación de outliers con IQR
- **Validación**: Puntuación automática de calidad
- **Categorización**: Clasificación automática de popularidad
- **Normalización**: Escalado de características

## 📋 Requisitos

### Librerías Principales
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
```

### Librerías de NLP
```
nltk>=3.7
spacy>=3.4.0
transformers>=4.20.0
torch>=1.12.0
textblob>=0.17.0
wordcloud>=1.8.0
```

### Instalación Automática
```python
from modules.environment_setup import setup_environment
setup_environment()  # Instala todo automáticamente
```

## 🎯 Ejemplos de Uso

### Análisis de Sentimientos Individual
```python
from modules.nlp_processor import analyze_sentiment

text = "This is an amazing and inspiring talk!"
result = analyze_sentiment(text)
print(f"Polaridad: {result['polarity']}")
print(f"Sentimiento: {result['sentiment_label']}")
```

### Crear Visualización Específica
```python
from modules.visualizer import create_wordcloud
import pandas as pd

df = pd.read_csv('ted_talks_en.csv')
create_wordcloud(df['transcript'], "Palabras más Comunes")
```

### Entrenar Solo un Modelo Específico
```python
from modules.ml_models import TedTalkClassifier

classifier = TedTalkClassifier()
X, y = classifier.prepare_features(df)
X_train, X_test, y_train, y_test = classifier.split_data(X, y)
classifier.train_models(X_train, y_train)
results = classifier.evaluate_models(X_test, y_test)
```

## 🔍 Análisis Incluidos

### 📊 Exploración de Datos
- Distribución de views y categorías de popularidad
- Análisis de longitud de transcripciones y títulos
- Correlaciones entre variables numéricas
- Detección y eliminación de outliers

### 🔤 Procesamiento de Texto
- Limpieza y normalización de texto
- Extracción de características textuales
- Análisis de sentimientos (polaridad y subjetividad)
- Identificación de entidades nombradas
- Análisis de frecuencia de palabras

### 🤖 Machine Learning
- Clasificación de popularidad en 5 categorías
- Entrenamiento de múltiples modelos
- Evaluación con métricas estándar
- Selección automática del mejor modelo
- Importancia de características

### 📈 Visualizaciones
- Gráficos de distribución y correlación
- Análisis de sentimientos por categoría
- Matrices de confusión de modelos
- Gráficos interactivos para exploración
- Nubes de palabras temáticas

## 🎯 Objetivos del Proyecto

1. **Modularidad**: Código organizado en módulos reutilizables
2. **Flexibilidad**: Uso individual de funciones o pipeline completo
3. **Profesionalidad**: Siguiendo estándares de la industria
4. **Rendimiento**: Optimizado para datasets grandes
5. **Interpretabilidad**: Resultados claros y visualizaciones informativas

## 📝 Notas Importantes

- **GPU Opcional**: El proyecto funciona en CPU, GPU mejora rendimiento
- **Memoria**: Recomendado 8GB+ RAM para datasets grandes
- **Tiempo**: El análisis completo puede tomar 10-30 minutos
- **Modularidad**: Cada módulo puede usarse independientemente
- **Extensibilidad**: Fácil añadir nuevos modelos o características

## 🤝 Contribuciones

Este proyecto está diseñado para ser extensible. Puedes:
- Añadir nuevos modelos de ML en `ml_models.py`
- Crear nuevas visualizaciones en `visualizer.py`
- Implementar técnicas de NLP en `nlp_processor.py`
- Mejorar la limpieza de datos en `data_cleaner.py`

---

¡Esperamos que esta arquitectura modular te ayude a realizar análisis más eficientes y mantenibles! 🚀
