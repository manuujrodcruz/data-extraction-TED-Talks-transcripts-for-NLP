"""
Módulo para modelos de machine learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           f1_score, accuracy_score, precision_score, 
                           recall_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class TedTalkClassifier:
    """
    Clase principal para clasificación de popularidad de TED Talks
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        self.encoders = {}
        self.feature_names = []
        self.results = {}
        
    def prepare_features(self, df, text_column='transcript_clean', target_column='popularity_numeric'):
        """
        Prepara las características para el modelado
        """
        print("=== PREPARANDO CARACTERÍSTICAS ===")
        
        # Características numéricas
        numeric_features = []
        
        # Características básicas
        if 'views' in df.columns:
            numeric_features.append('views')
        
        # Características de texto
        text_features = [col for col in df.columns if col.startswith('text_')]
        numeric_features.extend(text_features)
        
        # Características de sentimiento (solo numéricas)
        sentiment_features = [col for col in df.columns if col.startswith('sentiment_') 
                            and col not in ['sentiment_label', 'sentiment_sentiment_label']]
        numeric_features.extend(sentiment_features)
        
        # Características de entidades
        entity_features = [col for col in df.columns if col.endswith('_count')]
        numeric_features.extend(entity_features)
        
        # Filtrar columnas que existen en el DataFrame
        numeric_features = [col for col in numeric_features if col in df.columns]
        
        print(f"Características numéricas seleccionadas: {len(numeric_features)}")
        for feature in numeric_features:
            print(f"  - {feature}")
        
        # Crear matriz de características numéricas
        x_numeric = df[numeric_features].fillna(0)
        
        # Identificar y codificar características categóricas
        categorical_features = []
        if 'sentiment_label' in df.columns:
            categorical_features.append('sentiment_label')
        
        # Aplicar one-hot encoding a características categóricas
        x_categorical = None
        if categorical_features:
            print(f"Aplicando one-hot encoding a {len(categorical_features)} características categóricas:")
            for feature in categorical_features:
                print(f"  - {feature}")
            
            # Crear DataFrame con características categóricas
            categorical_data = df[categorical_features].fillna('unknown')
            
            # Aplicar one-hot encoding
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            x_categorical_encoded = encoder.fit_transform(categorical_data)
            
            # Crear nombres de columnas para las características codificadas
            categorical_feature_names = []
            for i, feature in enumerate(categorical_features):
                for category in encoder.categories_[i]:
                    categorical_feature_names.append(f"{feature}_{category}")
            
            # Convertir a DataFrame
            x_categorical = pd.DataFrame(
                x_categorical_encoded, 
                columns=categorical_feature_names, 
                index=x_numeric.index
            )
            
            print(f"One-hot encoding completado: {len(categorical_feature_names)} nuevas características")
            
            # Guardar el encoder para uso futuro
            self.encoders = {'categorical': encoder, 'categorical_features': categorical_features}
        
        # Características de texto con TF-IDF
        x_text = None
        if text_column in df.columns:
            print(f"\nCreando características TF-IDF de {text_column}...")
            
            # Limpiar texto
            text_data = df[text_column].fillna('').astype(str)
            
            # Vectorización TF-IDF
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8
            )
            
            x_text = self.vectorizers['tfidf'].fit_transform(text_data)
            print(f"Matriz TF-IDF: {x_text.shape[0]} muestras x {x_text.shape[1]} características")
            
            # Convertir TF-IDF a DataFrame
            tfidf_feature_names = [f'tfidf_{i}' for i in range(x_text.shape[1])]
            x_text_df = pd.DataFrame(x_text.toarray(), columns=tfidf_feature_names, index=x_numeric.index)
        
        # Combinar características
        if x_text is not None and x_categorical is not None:
            # Combinar numéricas, categóricas y TF-IDF
            x_combined = pd.concat([x_numeric, x_categorical, x_text_df], axis=1)
        elif x_text is not None:
            # Combinar numéricas y TF-IDF
            x_combined = pd.concat([x_numeric, x_text_df], axis=1)
        elif x_categorical is not None:
            # Combinar numéricas y categóricas
            x_combined = pd.concat([x_numeric, x_categorical], axis=1)
        else:
            # Solo numéricas
            x_combined = x_numeric
        
        # Target
        y = df[target_column]
        
        self.feature_names = list(x_combined.columns)
        
        print(f"\nMatriz final de características: {x_combined.shape[0]} muestras x {x_combined.shape[1]} características")
        print("Distribución del target:")
        print(y.value_counts().sort_index())
        
        return x_combined, y
    
    def split_data(self, x, y, test_size=0.2, random_state=42):
        """
        Divide los datos en entrenamiento y prueba
        """
        print("\n=== DIVIDIENDO DATOS ===")
        print(f"Tamaño del conjunto de prueba: {test_size*100}%")
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Conjunto de entrenamiento: {x_train.shape[0]} muestras")
        print(f"Conjunto de prueba: {x_test.shape[0]} muestras")
        
        # Escalar características numéricas (excluir TF-IDF y características categóricas codificadas)
        numeric_columns = [col for col in x.columns if not col.startswith('tfidf_') 
                          and not any(col.startswith(f"{feat}_") for feat in self.encoders.get('categorical_features', []))]
        if numeric_columns:
            self.scalers['standard'] = StandardScaler()
            x_train[numeric_columns] = self.scalers['standard'].fit_transform(x_train[numeric_columns])
            x_test[numeric_columns] = self.scalers['standard'].transform(x_test[numeric_columns])
            print(f"Escaladas {len(numeric_columns)} características numéricas")
        
        return x_train, x_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Entrena múltiples modelos de machine learning
        """
        print("\n=== ENTRENANDO MODELOS ===")
        
        # Definir modelos
        model_configs = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                C=1.0,
                kernel='rbf',
                gamma='scale'
            )
        }
        
        # Entrenar cada modelo
        for name, model in model_configs.items():
            print(f"\nEntrenando {name}...")
            
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                print(f"✓ {name} entrenado exitosamente")
                
                # Validación cruzada
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                print(f"  - CV F1-score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"✗ Error entrenando {name}: {e}")
        
        print(f"\n✓ {len(self.models)} modelos entrenados exitosamente")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evalúa todos los modelos entrenados
        """
        print("\n=== EVALUANDO MODELOS ===")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluando {name}...")
            
            try:
                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # AUC si es posible
                if y_pred_proba is not None and len(np.unique(y_test)) > 2:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                        results['auc'] = auc
                    except Exception:
                        results['auc'] = None
                
                evaluation_results[name] = results
                
                # Mostrar métricas
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-score: {f1:.4f}")
                if 'auc' in results and results['auc'] is not None:
                    print(f"  AUC: {results['auc']:.4f}")
                
            except Exception as e:
                print(f"✗ Error evaluando {name}: {e}")
                evaluation_results[name] = None
        
        self.results = evaluation_results
        return evaluation_results
    
    def create_evaluation_plots(self, y_test):
        """
        Crea visualizaciones de evaluación de modelos
        """
        if not self.results:
            print("No hay resultados para visualizar")
            return
        
        # Filtrar modelos con resultados válidos
        valid_results = {name: results for name, results in self.results.items() if results is not None}
        
        if not valid_results:
            print("No hay resultados válidos para visualizar")
            return
        
        # 1. Comparación de métricas
        metrics_df = []
        for name, results in valid_results.items():
            metrics_df.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        metrics_df = pd.DataFrame(metrics_df)
        
        # Gráfico de barras de métricas
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics_df))
        width = 0.2
        
        plt.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', alpha=0.8)
        plt.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        plt.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        plt.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Modelos')
        plt.ylabel('Puntuación')
        plt.title('Comparación de Métricas por Modelo')
        plt.xticks(x, metrics_df['Model'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Matrices de confusión
        n_models = len(valid_results)
        rows = (n_models + 1) // 2
        
        _, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(valid_results.items()):
            row = i // 2
            col = i % 2
            
            cm = confusion_matrix(y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[row][col], cmap='Blues')
            axes[row][col].set_title(f'Matriz de Confusión - {name}')
            axes[row][col].set_xlabel('Predicción')
            axes[row][col].set_ylabel('Real')
        
        # Ocultar subplot vacío si es impar
        if n_models % 2 == 1:
            axes[-1][-1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, model_name='Random Forest', top_n=20):
        """
        Obtiene la importancia de características para modelos que la soportan
        """
        if model_name not in self.models:
            print(f"Modelo {model_name} no encontrado")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Crear DataFrame con importancias
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} características más importantes ({model_name}):")
            top_features = feature_importance_df.head(top_n)
            
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Visualización
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), top_features['importance'].iloc[::-1])
            plt.yticks(range(top_n), top_features['feature'].iloc[::-1])
            plt.xlabel('Importancia')
            plt.title(f'Top {top_n} Características Más Importantes - {model_name}')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        
        else:
            print(f"El modelo {model_name} no soporta feature importance")
            return None
    
    def get_best_model(self, metric='f1_score'):
        """
        Obtiene el mejor modelo según la métrica especificada
        """
        if not self.results:
            print("No hay resultados disponibles")
            return None
        
        valid_results = {name: results for name, results in self.results.items() 
                        if results is not None and metric in results}
        
        if not valid_results:
            print(f"No hay resultados válidos para la métrica {metric}")
            return None
        
        best_model_name = max(valid_results.keys(), 
                             key=lambda x: valid_results[x][metric])
        best_score = valid_results[best_model_name][metric]
        
        print(f"\nMejor modelo según {metric}: {best_model_name}")
        print(f"Puntuación: {best_score:.4f}")
        
        return best_model_name, self.models[best_model_name], best_score
    
    def predict_new_data(self, x_new, model_name=None):
        """
        Realiza predicciones en nuevos datos
        """
        if model_name is None:
            model_name, _, _ = self.get_best_model()
            if model_name is None:
                print("No hay modelos disponibles")
                return None
        
        if model_name not in self.models:
            print(f"Modelo {model_name} no encontrado")
            return None
        
        model = self.models[model_name]
        
        # Aplicar transformaciones si es necesario
        x_scaled = x_new.copy()
        
        # Aplicar encoding categórico si existe
        if 'categorical' in self.encoders:
            categorical_features = self.encoders['categorical_features']
            if any(col in x_new.columns for col in categorical_features):
                encoder = self.encoders['categorical']
                categorical_data = x_new[categorical_features].fillna('unknown')
                x_categorical_encoded = encoder.transform(categorical_data)
                
                # Crear nombres de columnas para las características codificadas
                categorical_feature_names = []
                for i, feature in enumerate(categorical_features):
                    for category in encoder.categories_[i]:
                        categorical_feature_names.append(f"{feature}_{category}")
                
                # Agregar características codificadas
                x_categorical_df = pd.DataFrame(
                    x_categorical_encoded, 
                    columns=categorical_feature_names, 
                    index=x_scaled.index
                )
                x_scaled = pd.concat([x_scaled, x_categorical_df], axis=1)
                
                # Eliminar columnas categóricas originales
                x_scaled = x_scaled.drop(columns=categorical_features, errors='ignore')
        
        # Escalar características numéricas
        numeric_columns = [col for col in x_scaled.columns if not col.startswith('tfidf_') and not any(col.startswith(f"{feat}_") for feat in self.encoders.get('categorical_features', []))]
        if numeric_columns and 'standard' in self.scalers:
            x_scaled[numeric_columns] = self.scalers['standard'].transform(x_scaled[numeric_columns])
        
        # Predicciones
        predictions = model.predict(x_scaled)
        probabilities = model.predict_proba(x_scaled) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def save_model_summary(self, filename='model_summary.txt'):
        """
        Guarda un resumen de los resultados del modelo
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== RESUMEN DE MODELOS TED TALKS ===\n\n")
            
            if self.results:
                f.write("RESULTADOS DE EVALUACIÓN:\n")
                f.write("-" * 50 + "\n")
                
                for name, results in self.results.items():
                    if results is not None:
                        f.write(f"\n{name}:\n")
                        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                        f.write(f"  Precision: {results['precision']:.4f}\n")
                        f.write(f"  Recall: {results['recall']:.4f}\n")
                        f.write(f"  F1-Score: {results['f1_score']:.4f}\n")
                        if 'auc' in results and results['auc'] is not None:
                            f.write(f"  AUC: {results['auc']:.4f}\n")
            
            # Mejor modelo
            best_model_info = self.get_best_model()
            if best_model_info[0] is not None:
                f.write(f"\nMEJOR MODELO: {best_model_info[0]}\n")
                f.write(f"F1-Score: {best_model_info[2]:.4f}\n")
            
            f.write(f"\nCARACTERÍSTICAS UTILIZADAS: {len(self.feature_names)}\n")
            f.write("MODELOS ENTRENADOS: " + ", ".join(self.models.keys()) + "\n")
        
        print(f"Resumen guardado en {filename}")


def create_ml_pipeline(df, text_column='transcript_clean', target_column='popularity_numeric'):
    """
    Función principal para crear y ejecutar el pipeline de ML
    """
    print("=== INICIANDO PIPELINE DE MACHINE LEARNING ===")
    
    # Crear instancia del clasificador
    classifier = TedTalkClassifier()
    
    # Preparar características
    X, y = classifier.prepare_features(df, text_column, target_column)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    
    # Entrenar modelos
    classifier.train_models(X_train, y_train)
    
    # Evaluar modelos
    results = classifier.evaluate_models(X_test, y_test)
    
    # Crear visualizaciones
    classifier.create_evaluation_plots(y_test)
    
    # Mostrar importancia de características
    classifier.get_feature_importance()
    
    # Obtener mejor modelo
    classifier.get_best_model()
    
    # Guardar resumen
    classifier.save_model_summary()
    
    print("\n✓ Pipeline de Machine Learning completado")
    
    return classifier, results
