"""
Módulo para visualización de datos y resultados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter

# Constantes
POPULARITY_CATEGORY_LABEL = 'Categoría de Popularidad'


def setup_plot_style():
    """Configura el estilo de los gráficos"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def create_data_overview_plots(df):
    """
    Crea visualizaciones de resumen del dataset
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis Exploratorio del Dataset', fontsize=16, fontweight='bold')

    # 1. Distribución de Views
    axes[0, 0].hist(df['views'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribución de Views')
    axes[0, 0].set_xlabel('Número de Views')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3)

    # Estadísticas
    mean_views = df['views'].mean()
    median_views = df['views'].median()
    axes[0, 0].axvline(mean_views, color='red', linestyle='--', label=f'Media: {mean_views:,.0f}')
    axes[0, 0].axvline(median_views, color='green', linestyle='--', label=f'Mediana: {median_views:,.0f}')
    axes[0, 0].legend()

    # 2. Distribución de Categorías de Popularidad
    if 'popularity_category' in df.columns:
        category_counts = df['popularity_category'].value_counts().sort_index()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        bars = axes[0, 1].bar(category_counts.index, category_counts.values, color=colors, alpha=0.8)
        axes[0, 1].set_title('Distribución de Categorías de Popularidad')
        axes[0, 1].set_xlabel('Categoría')
        axes[0, 1].set_ylabel('Número de Videos')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Porcentajes en las barras
        for bar, count in zip(bars, category_counts.values):
            height = bar.get_height()
            percentage = (count / len(df)) * 100
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                           f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. Longitud de Transcripciones
    if 'transcript_clean' in df.columns:
        transcript_lengths = df['transcript_clean'].str.len()
        axes[1, 0].hist(transcript_lengths, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_title('Distribución de Longitud de Transcripciones')
        axes[1, 0].set_xlabel('Número de Caracteres')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(True, alpha=0.3)
        
        mean_length = transcript_lengths.mean()
        axes[1, 0].axvline(mean_length, color='red', linestyle='--', 
                          label=f'Media: {mean_length:.0f} caracteres')
        axes[1, 0].legend()

    # 4. Relación Views vs Longitud de Título
    if 'title_clean' in df.columns:
        title_lengths = df['title_clean'].str.len()
        axes[1, 1].scatter(title_lengths, df['views'], alpha=0.6, 
                          color='mediumpurple', s=30)
        axes[1, 1].set_title('Relación: Longitud del Título vs Views')
        axes[1, 1].set_xlabel('Longitud del Título (caracteres)')
        axes[1, 1].set_ylabel('Views')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Línea de tendencia
        z = np.polyfit(title_lengths, df['views'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(title_lengths, p(title_lengths), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(df, numeric_columns=None):
    """
    Crea un mapa de calor de correlaciones
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        print("No hay suficientes columnas numéricas para crear correlaciones")
        return
    
    correlation_matrix = df[numeric_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_sentiment_analysis_plots(df):
    """
    Crea visualizaciones específicas para análisis de sentimientos
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Sentimientos', fontsize=16, fontweight='bold')
    print(df.columns.tolist())
    
    # 1. Distribución de Polaridad
    if 'sentiment_polarity' in df.columns:
        axes[0, 0].hist(df['sentiment_polarity'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de Polaridad de Sentimientos')
        axes[0, 0].set_xlabel('Polaridad (-1 = Negativo, +1 = Positivo)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Neutral')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # 2. Distribución de Subjetividad
    if 'sentiment_subjectivity' in df.columns:
        axes[0, 1].hist(df['sentiment_subjectivity'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribución de Subjetividad')
        axes[0, 1].set_xlabel('Subjetividad (0 = Objetivo, 1 = Subjetivo)')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sentimiento por Categoría de Popularidad
    if 'sentiment_sentiment_label' in df.columns and 'popularity_category' in df.columns:
        sentiment_popularity = pd.crosstab(df['popularity_category'], df['sentiment_sentiment_label'])
        sentiment_popularity_pct = sentiment_popularity.div(sentiment_popularity.sum(axis=1), axis=0) * 100
        
        sentiment_popularity_pct.plot(kind='bar', ax=axes[1, 0], stacked=True, alpha=0.8)
        axes[1, 0].set_title(f'Distribución de Sentimientos por {POPULARITY_CATEGORY_LABEL} (%)')
        axes[1, 0].set_xlabel(POPULARITY_CATEGORY_LABEL)
        axes[1, 0].set_ylabel('Porcentaje')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Sentimiento')
    
    # 4. Relación Views vs Polaridad
    if 'sentiment_polarity' in df.columns:
        axes[1, 1].scatter(df['sentiment_polarity'], df['views'], alpha=0.6, color='orange')
        axes[1, 1].set_title('Relación: Polaridad de Sentimiento vs Views')
        axes[1, 1].set_xlabel('Polaridad de Sentimiento')
        axes[1, 1].set_ylabel('Views')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Línea de tendencia
        if not df['sentiment_polarity'].isna().all():
            z = np.polyfit(df['sentiment_polarity'].dropna(), 
                          df.loc[df['sentiment_polarity'].notna(), 'views'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df['sentiment_polarity'].min(), df['sentiment_polarity'].max(), 100)
            axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.show()


def create_text_features_plots(df):
    """
    Crea visualizaciones para características textuales
    """
    setup_plot_style()
    
    text_feature_columns = [col for col in df.columns if col.startswith('text_')]
    
    if not text_feature_columns:
        print("No se encontraron características textuales para visualizar")
        return
    
    # Seleccionar características más relevantes
    key_features = ['text_word_count', 'text_sentence_count', 'text_lexical_diversity', 'text_avg_word_length']
    available_features = [col for col in key_features if col in df.columns]
    
    if not available_features:
        available_features = text_feature_columns[:4]  # Tomar las primeras 4
    
    n_features = len(available_features)
    rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6*rows))
    if rows == 1:
        axes = [axes]
    
    fig.suptitle('Análisis de Características Textuales', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(available_features):
        row = i // 2
        col = i % 2
        
        # Histograma de la característica
        axes[row][col].hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
        axes[row][col].set_title(f'Distribución de {feature.replace("text_", "").replace("_", " ").title()}')
        axes[row][col].set_xlabel(feature.replace("text_", "").replace("_", " ").title())
        axes[row][col].set_ylabel('Frecuencia')
        axes[row][col].grid(True, alpha=0.3)
        
        # Estadísticas
        mean_val = df[feature].mean()
        median_val = df[feature].median()
        axes[row][col].axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
        axes[row][col].axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
        axes[row][col].legend()
    
    # Ocultar subplot vacío si es impar
    if n_features % 2 == 1:
        axes[-1][-1].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def create_entity_analysis_plots(df):
    """
    Crea visualizaciones para análisis de entidades nombradas
    """
    entity_columns = [col for col in df.columns if col.endswith('_count')]
    
    if not entity_columns:
        print("No se encontraron columnas de entidades para visualizar")
        return
    
    setup_plot_style()
    
    # Resumen de entidades
    entity_totals = {}
    for col in entity_columns:
        entity_type = col.replace('_count', '').upper()
        entity_totals[entity_type] = df[col].sum()
    
    # Gráfico de barras de entidades totales
    plt.figure(figsize=(12, 6))
    entity_types = list(entity_totals.keys())
    entity_counts = list(entity_totals.values())
    
    plt.bar(entity_types, entity_counts, alpha=0.8, color='skyblue')
    plt.title('Total de Entidades Nombradas por Tipo')
    plt.xlabel('Tipo de Entidad')
    plt.ylabel('Cantidad Total')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, count in enumerate(entity_counts):
        plt.text(i, count + max(entity_counts)*0.01, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Boxplot de entidades por categoría de popularidad
    if 'popularity_category' in df.columns and entity_columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Distribución de Entidades por Categoría de Popularidad', fontsize=16)
        
        # Seleccionar las 4 entidades más comunes
        top_entities = sorted(entity_totals.items(), key=lambda x: x[1], reverse=True)[:4]
        
        for i, (entity_type, _) in enumerate(top_entities):
            row = i // 2
            col = i % 2
            
            entity_col = f'{entity_type.lower()}_count'
            if entity_col in df.columns:
                df.boxplot(column=entity_col, by='popularity_category', ax=axes[row][col])
                axes[row][col].set_title(f'Distribución de {entity_type} por Popularidad')
                axes[row][col].set_xlabel(POPULARITY_CATEGORY_LABEL)
                axes[row][col].set_ylabel(f'Cantidad de {entity_type}')
        
        plt.tight_layout()
        plt.show()


def create_wordcloud(text_data, title="Word Cloud", max_words=100):
    """
    Crea una nube de palabras
    """
    if isinstance(text_data, pd.Series):
        text = ' '.join(text_data.fillna('').astype(str))
    else:
        text = str(text_data)
    
    if not text.strip():
        print("No hay texto suficiente para crear la nube de palabras")
        return
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=max_words,
                         colormap='viridis').generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_interactive_plots(df):
    """
    Crea gráficos interactivos con Plotly
    """
    # 1. Scatter plot interactivo: Views vs características textuales
    if 'text_word_count' in df.columns:
        fig1 = px.scatter(df, 
                         x='text_word_count', 
                         y='views',
                         color='popularity_category' if 'popularity_category' in df.columns else None,
                         title='Relación entre Cantidad de Palabras y Views',
                         labels={'text_word_count': 'Cantidad de Palabras', 'views': 'Views'},
                         hover_data=['title_clean'] if 'title_clean' in df.columns else None)
        fig1.show()
    
    # 2. Box plot interactivo: Sentimientos por categoría
    if 'sentiment_polarity' in df.columns and 'popularity_category' in df.columns:
        fig2 = px.box(df, 
                     x='popularity_category', 
                     y='sentiment_polarity',
                     title='Distribución de Polaridad de Sentimiento por Categoría de Popularidad',
                     labels={'popularity_category': POPULARITY_CATEGORY_LABEL, 
                            'sentiment_polarity': 'Polaridad de Sentimiento'})
        fig2.show()
    
    # 3. Histograma interactivo de views
    fig3 = px.histogram(df, 
                       x='views', 
                       nbins=50,
                       title='Distribución de Views (Interactivo)',
                       labels={'views': 'Views', 'count': 'Frecuencia'})
    fig3.show()


def print_summary_statistics(df):
    """
    Imprime estadísticas resumidas del dataset
    """
    print("=== ESTADÍSTICAS RESUMIDAS ===")
    print(f"Total de videos: {len(df):,}")
    
    if 'views' in df.columns:
        print(f"Promedio de views: {df['views'].mean():,.0f}")
        print(f"Mediana de views: {df['views'].median():,.0f}")
        print(f"Desviación estándar: {df['views'].std():,.0f}")
    
    # Características textuales
    if 'transcript_clean' in df.columns:
        avg_transcript_length = df['transcript_clean'].str.len().mean()
        print(f"Longitud promedio de transcripción: {avg_transcript_length:.0f} caracteres")
    
    if 'title_clean' in df.columns:
        avg_title_length = df['title_clean'].str.len().mean()
        print(f"Longitud promedio de título: {avg_title_length:.1f} caracteres")
    
    # Sentimientos
    if 'sentiment_polarity' in df.columns:
        avg_polarity = df['sentiment_polarity'].mean()
        print(f"Polaridad promedio de sentimiento: {avg_polarity:.3f}")
    
    # Distribución de categorías
    if 'popularity_category' in df.columns:
        print("\nDistribución de categorías de popularidad:")
        category_dist = df['popularity_category'].value_counts().sort_index()
        for category, count in category_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
