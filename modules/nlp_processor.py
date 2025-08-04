"""
Módulo para procesamiento de lenguaje natural (NLP)
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
from tqdm.auto import tqdm
from .progress_tracker import ProgressTracker, real_time_feedback

try:
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    print(f"Warning: Some NLP libraries not available: {e}")


def load_nlp_models():
    """Carga modelos de NLP necesarios"""
    models = {}
    
    # Cargar spaCy
    try:
        models['spacy'] = spacy.load("en_core_web_sm")
        print("✓ spaCy cargado correctamente")
    except Exception as e:
        print(f"⚠ Error cargando spaCy: {e}")
        models['spacy'] = None
    
    # Configurar NLTK
    try:
        models['stop_words'] = set(stopwords.words('english'))
        models['lemmatizer'] = WordNetLemmatizer()
        print("✓ NLTK configurado correctamente")
    except Exception as e:
        print(f"⚠ Error configurando NLTK: {e}")
        models['stop_words'] = set()
        models['lemmatizer'] = None
    
    return models


def extract_named_entities(text, nlp_model):
    """
    Extrae entidades nombradas usando spaCy
    """
    if pd.isna(text) or text == '' or nlp_model is None:
        return {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'MONEY': [],
            'DATE': [],
            'TIME': [],
            'PERCENT': [],
            'QUANTITY': []
        }
    
    # Limitar texto para eficiencia
    text_limited = text[:1000000] if len(text) > 1000000 else text
    doc = nlp_model(text_limited)
    
    entities = {
        'PERSON': [],
        'ORG': [],
        'GPE': [],      # Geopolitical entities
        'MONEY': [],
        'DATE': [],
        'TIME': [],
        'PERCENT': [],
        'QUANTITY': []
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text.lower().strip())
    
    # Eliminar duplicados manteniendo orden
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return entities


def count_entity_types(entities_dict):
    """
    Cuenta el número de entidades por tipo
    """
    counts = {}
    for ent_type, ent_list in entities_dict.items():
        counts[f'{ent_type.lower()}_count'] = len(ent_list)
    
    return counts


def analyze_sentiment(text):
    """
    Analiza el sentimiento del texto usando TextBlob
    """
    if pd.isna(text) or text == '':
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'neutral'
        }
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Clasificar sentimiento
        if polarity > 0.1:
            sentiment_label = 'positive'
        elif polarity < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_label': sentiment_label
        }
    
    except Exception as e:
        print(f"Error analizando sentimiento: {e}")
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'neutral'
        }


def extract_text_features(text, stop_words=None):
    """
    Extrae características textuales básicas
    """
    if pd.isna(text) or text == '':
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'unique_words': 0,
            'lexical_diversity': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0
        }
    
    # Contar palabras y oraciones
    words = word_tokenize(text.lower()) if 'word_tokenize' in globals() else text.split()
    sentences = sent_tokenize(text) if 'sent_tokenize' in globals() else text.split('.')
    
    # Filtrar stop words si están disponibles
    if stop_words:
        words = [word for word in words if word not in stop_words and word.isalpha()]
    else:
        words = [word for word in words if word.isalpha()]
    
    # Calcular métricas
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Contar signos de puntuación
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Ratio de mayúsculas
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'unique_words': unique_words,
        'lexical_diversity': lexical_diversity,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': uppercase_ratio
    }


def extract_keywords(text, n_keywords=10, stop_words=None):
    """
    Extrae palabras clave más frecuentes del texto
    """
    if pd.isna(text) or text == '':
        return []
    
    # Limpiar y tokenizar
    words = word_tokenize(text.lower()) if 'word_tokenize' in globals() else text.lower().split()
    
    # Filtrar palabras
    if stop_words:
        words = [word for word in words if word not in stop_words and word.isalpha() and len(word) > 3]
    else:
        words = [word for word in words if word.isalpha() and len(word) > 3]
    
    # Contar frecuencias
    word_freq = Counter(words)
    
    # Retornar top keywords
    return word_freq.most_common(n_keywords)


def process_text_features(df, text_column='transcript_clean', nlp_models=None):
    """
    Procesa todas las características de texto para un DataFrame
    """
    # Inicializar tracker de progreso
    tracker = ProgressTracker(total_steps=5, description="Procesamiento NLP")
    tracker.start("Iniciando extracción de características NLP")
    
    real_time_feedback(f"Procesando columna: {text_column}")
    
    if text_column not in df.columns:
        print(f"⚠ Columna {text_column} no encontrada")
        return df
    
    # 1. CARGAR MODELOS
    tracker.step("Cargando modelos de NLP")
    if nlp_models is None:
        nlp_models = load_nlp_models()
        real_time_feedback("Modelos de NLP cargados")
    
    # 2. PREPARAR MUESTRA
    tracker.step("Preparando muestra de datos")
    sample_size = min(100, len(df))
    real_time_feedback(f"Procesando muestra de {sample_size} textos para velocidad...")
    sample_df = df.head(sample_size).copy()
    
    # 3. PROCESAR SENTIMIENTOS
    tracker.step("Analizando sentimientos con TextBlob")
    sample_df = _process_sentiments(sample_df, text_column)
    real_time_feedback("Análisis de sentimientos completado")
    
    # 4. PROCESAR CARACTERÍSTICAS TEXTUALES
    tracker.step("Extrayendo características textuales")
    sample_df = _process_text_features(sample_df, text_column, nlp_models)
    real_time_feedback("Características textuales extraídas")
    
    # 5. PROCESAR ENTIDADES NOMBRADAS
    tracker.step("Identificando entidades nombradas")
    if nlp_models['spacy'] is not None:
        sample_df = _process_named_entities(sample_df, text_column, nlp_models['spacy'])
        real_time_feedback("Entidades nombradas identificadas")
    else:
        real_time_feedback("spaCy no disponible - omitiendo entidades nombradas")
    
    # Mostrar estadísticas finales
    _show_feature_statistics(sample_df)
    
    tracker.finish("Procesamiento NLP completado")
    
    return sample_df


def _process_sentiments(df, text_column):
    """Procesa análisis de sentimientos"""
    real_time_feedback("Analizando polaridad y subjetividad...")
    tqdm.pandas(desc="Sentimientos", leave=False)
    sentiment_results = df[text_column].progress_apply(analyze_sentiment)
    
    # Convertir resultados a columnas
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    for col in sentiment_df.columns:
        df[f'sentiment_{col}'] = sentiment_df[col]
    
    return df


def _process_text_features(df, text_column, nlp_models):
    """Procesa características textuales básicas"""
    real_time_feedback("Calculando longitud, palabras, oraciones...")
    tqdm.pandas(desc="Características", leave=False)
    text_features = df[text_column].progress_apply(
        lambda x: extract_text_features(x, nlp_models['stop_words'])
    )
    
    # Convertir a columnas
    features_df = pd.DataFrame(text_features.tolist())
    for col in features_df.columns:
        df[f'text_{col}'] = features_df[col]
    
    return df


def _process_named_entities(df, text_column, spacy_model):
    """Procesa entidades nombradas"""
    entity_sample_size = min(50, len(df))
    real_time_feedback(f"Procesando {entity_sample_size} textos para entidades nombradas...")
    
    tqdm.pandas(desc="Entidades", leave=False)
    entity_results = df[text_column].head(entity_sample_size).progress_apply(
        lambda x: extract_named_entities(x, spacy_model)
    )
    
    # Procesar conteos de entidades
    entity_counts_list = []
    for entities in entity_results:
        counts = count_entity_types(entities)
        entity_counts_list.append(counts)
    
    # Añadir al dataframe
    entity_df = pd.DataFrame(entity_counts_list)
    
    # Inicializar columnas de entidades para toda la muestra
    for col in entity_df.columns:
        df[col] = 0
    
    # Llenar valores para las filas procesadas
    for i, (_, row) in enumerate(entity_df.iterrows()):
        for col, value in row.items():
            df.loc[i, col] = value
    
    return df


def _show_feature_statistics(df):
    """Muestra estadísticas de las características procesadas"""
    print("\n=== ESTADÍSTICAS DE CARACTERÍSTICAS ===")
    
    # Sentimientos
    if 'sentiment_polarity' in df.columns:
        polarity_stats = df['sentiment_polarity'].describe()
        print("Polaridad de sentimiento:")
        print(f"  Media: {polarity_stats['mean']:.3f}")
        print(f"  Rango: [{polarity_stats['min']:.3f}, {polarity_stats['max']:.3f}]")
        
        sentiment_dist = df['sentiment_sentiment_label'].value_counts()
        print("Distribución de sentimientos:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    # Características textuales
    text_feature_columns = [col for col in df.columns if col.startswith('text_')]
    if text_feature_columns:
        print("\nCaracterísticas textuales promedio:")
        for col in text_feature_columns:
            mean_val = df[col].mean()
            print(f"  {col.replace('text_', '')}: {mean_val:.2f}")
    
    # Entidades
    entity_columns = [col for col in df.columns if col.endswith('_count')]
    if entity_columns:
        print("\nEntidades promedio por texto:")
        for col in entity_columns:
            mean_val = df[col].mean()
            entity_type = col.replace('_count', '').upper()
            print(f"  {entity_type}: {mean_val:.2f}")


def get_text_statistics(df, text_columns=None):
    """
    Obtiene estadísticas generales de las columnas de texto
    """
    if text_columns is None:
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    stats = {}
    
    for col in text_columns:
        if col in df.columns:
            col_stats = {
                'total_texts': len(df),
                'non_empty_texts': df[col].notna().sum(),
                'avg_length': df[col].str.len().mean(),
                'median_length': df[col].str.len().median(),
                'max_length': df[col].str.len().max(),
                'min_length': df[col].str.len().min()
            }
            stats[col] = col_stats
    
    return stats


def create_word_frequency_analysis(df, text_column, top_n=20, stop_words=None):
    """
    Crea análisis de frecuencia de palabras para todo el corpus
    """
    print("=== ANÁLISIS DE FRECUENCIA DE PALABRAS ===")
    
    if text_column not in df.columns:
        print(f"⚠ Columna {text_column} no encontrada")
        return {}
    
    # Combinar todos los textos
    all_text = ' '.join(df[text_column].fillna('').astype(str))
    
    # Extraer palabras clave del corpus completo
    keywords = extract_keywords(all_text, n_keywords=top_n, stop_words=stop_words)
    
    print(f"Top {top_n} palabras más frecuentes:")
    for word, freq in keywords:
        print(f"  {word}: {freq}")
    
    return dict(keywords)
