"""
Módulo para limpieza profesional de datos
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from .progress_tracker import ProgressTracker, real_time_feedback


def clean_text(text):
    """Función para limpiar texto"""
    if pd.isna(text) or text == '' or text == 'nan':
        return ''
    
    text = str(text).lower()
    # Eliminar caracteres especiales pero mantener espacios y puntuación básica
    text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text


def remove_outliers_iqr(df, column):
    """Elimina outliers usando el método IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    print(f"   - Q1 (25%): {Q1:,.0f}")
    print(f"   - Q3 (75%): {Q3:,.0f}")
    print(f"   - IQR: {IQR:,.0f}")
    print(f"   - Límite inferior: {lower_bound:,.0f}")
    print(f"   - Límite superior: {upper_bound:,.0f}")
    print(f"   - Outliers identificados: {outliers_count} ({(outliers_count/len(df)*100):.2f}%)")
    
    return df[~outliers_mask].reset_index(drop=True), outliers_count


def create_popularity_categories(df, views_column='views'):
    """Crea categorías de popularidad basadas en percentiles"""
    # Calcular percentiles
    percentiles = [0, 20, 40, 60, 80, 100]
    thresholds = [df[views_column].quantile(p/100) for p in percentiles]
    
    print("   Umbrales de popularidad:")
    categories = ['Bajo', 'Medio Bajo', 'Medio', 'Medio Alto', 'Alto']
    
    for i, (cat, threshold) in enumerate(zip(categories, thresholds[1:])):
        print(f"     - {cat}: hasta {threshold:,.0f} views")
    
    # Crear categorías
    df['popularity_category'] = pd.cut(
        df[views_column], 
        bins=thresholds, 
        labels=categories,
        include_lowest=True
    )
    
    # Crear variable numérica para modelado
    label_encoder = LabelEncoder()
    df['popularity_numeric'] = label_encoder.fit_transform(df['popularity_category'])
    
    # Mostrar distribución
    distribution = df['popularity_category'].value_counts().sort_index()
    print("\n   Distribución de categorías:")
    for cat, count in distribution.items():
        percentage = (count / len(df)) * 100
        print(f"     - {cat}: {count} ({percentage:.1f}%)")
    
    return df, label_encoder


def calculate_data_quality(df):
    """Calcula una puntuación de calidad de datos"""
    score = 0
    
    # Porcentaje de valores no nulos (peso: 3)
    non_null_percentage = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 3
    score += non_null_percentage
    
    # Diversidad en columnas categóricas (peso: 2)
    if 'popularity_category' in df.columns:
        category_balance = 1 - df['popularity_category'].value_counts().std() / df['popularity_category'].value_counts().mean()
        score += category_balance * 2
    
    # Consistencia en datos numéricos (peso: 2)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        consistency_score = 1 - df[numeric_columns].isnull().sum().sum() / (df.shape[0] * len(numeric_columns))
        score += consistency_score * 2
    
    # Longitud promedio de texto (peso: 1)
    text_columns = [col for col in df.columns if 'clean' in col]
    if text_columns:
        avg_text_length = df[text_columns[0]].str.len().mean()
        text_score = min(avg_text_length / 100, 1)  # Normalizar a [0,1]
        score += text_score
    
    # Variabilidad en views (peso: 2)
    if 'views' in df.columns:
        cv = df['views'].std() / df['views'].mean()  # Coeficiente de variación
        variability_score = min(cv / 2, 1)  # Normalizar
        score += variability_score * 2
    
    return min(score, 10)  # Máximo 10


def clean_dataset_professional(df):
    """
    Pipeline de limpieza profesional siguiendo estándares de la industria
    """
    df_clean = df.copy()
    cleaning_log = []
    
    # Inicializar tracker de progreso
    tracker = ProgressTracker(total_steps=4, description="Limpieza de datos")
    tracker.start("Iniciando limpieza profesional de datos")
    
    real_time_feedback(f"Dataset original: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")
    
    # 1. ELIMINACIÓN DE VALORES ATÍPICOS USANDO IQR
    tracker.step("Eliminando outliers con método IQR")
    
    if 'views' in df_clean.columns:
        print("   📊 Analizando distribución de 'views'...")
        df_clean, outliers_count = remove_outliers_iqr(df_clean, 'views')
        cleaning_log.append(f"Eliminados {outliers_count} outliers en 'views'")
        real_time_feedback(f"Dataset después de eliminar outliers: {df_clean.shape[0]} filas")
    
    # 2. LIMPIEZA DE DATOS TEXTUALES
    tracker.step("Limpiando datos textuales")
    
    text_columns = ['title', 'description', 'transcript', 'speaker', 'main_speaker']
    processed_columns = 0
    
    for col in text_columns:
        if col in df_clean.columns:
            real_time_feedback(f"Procesando columna: {col}")
            
            # Convertir a string y manejar nulos
            df_clean[col] = df_clean[col].astype(str)
            df_clean[col] = df_clean[col].replace('nan', '')
            
            # Limpiar texto
            df_clean[f'{col}_clean'] = df_clean[col].apply(clean_text)
            
            # Estadísticas
            null_count = df_clean[col].isin(['', 'nan', None]).sum()
            avg_length = df_clean[f'{col}_clean'].str.len().mean()
            
            print(f"     - Valores vacíos: {null_count}")
            print(f"     - Longitud promedio: {avg_length:.1f} caracteres")
            
            cleaning_log.append(f"Limpiado columna {col}: {null_count} valores vacíos")
            processed_columns += 1
    
    real_time_feedback(f"Procesadas {processed_columns} columnas de texto")
    
    # 3. CREACIÓN DE CATEGORÍAS DE POPULARIDAD
    tracker.step("Creando categorías de popularidad")
    
    if 'views' in df_clean.columns:
        print("   📈 Analizando distribución de popularidad...")
        df_clean, _ = create_popularity_categories(df_clean)
        cleaning_log.append("Creadas categorías de popularidad")
    
    # 4. VALIDACIÓN FINAL
    tracker.step("Validando dataset limpio")
    
    real_time_feedback(f"Dimensiones finales: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")
    rows_removed = len(df) - len(df_clean)
    removal_percentage = (rows_removed/len(df)*100)
    real_time_feedback(f"Filas eliminadas: {rows_removed} ({removal_percentage:.2f}%)")
    
    # Verificar calidad de datos
    quality_score = calculate_data_quality(df_clean)
    real_time_feedback(f"Puntuación de calidad: {quality_score:.2f}/10")
    
    tracker.finish("Limpieza de datos completada")
    
    return df_clean, cleaning_log


def get_data_summary(df):
    """Genera un resumen completo del dataset"""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
    }
    
    # Estadísticas de views si existe
    if 'views' in df.columns:
        summary['views_stats'] = {
            'min': df['views'].min(),
            'max': df['views'].max(),
            'mean': df['views'].mean(),
            'median': df['views'].median(),
            'std': df['views'].std()
        }
    
    return summary


def validate_data_quality(df, min_quality_score=7.0):
    """Valida la calidad del dataset limpio"""
    quality_score = calculate_data_quality(df)
    
    validation_results = {
        'quality_score': quality_score,
        'passed': quality_score >= min_quality_score,
        'issues': []
    }
    
    # Verificar problemas específicos
    if df.isnull().sum().sum() > 0:
        validation_results['issues'].append("Datos nulos presentes")
    
    if 'views' in df.columns and df['views'].min() < 0:
        validation_results['issues'].append("Views negativos encontrados")
    
    # Verificar balance de clases
    if 'popularity_category' in df.columns:
        category_counts = df['popularity_category'].value_counts()
        if category_counts.std() / category_counts.mean() > 0.5:
            validation_results['issues'].append("Desbalance significativo en categorías")
    
    return validation_results
