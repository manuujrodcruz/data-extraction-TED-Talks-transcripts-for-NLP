"""
Módulo para configuración del ambiente y instalación de dependencias
"""

import subprocess
import sys
import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def install_package(package):
    """Instala un paquete usando pip si no está disponible"""
    try:
        package_name = package.split('==')[0].split('[')[0]
        __import__(package_name)
        print(f"✓ {package} ya está instalado")
        return True
    except ImportError:
        try:
            print(f"Instalando {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=60  # Timeout de 60 segundos
            )
            if result.returncode == 0:
                print(f"✓ {package} instalado correctamente")
                return True
            else:
                print(f"⚠ Error instalando {package}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⚠ Timeout instalando {package}")
            return False
        except Exception as e:
            print(f"⚠ Error instalando {package}: {e}")
            return False

def setup_environment():
    """Configura el ambiente completo del proyecto"""
    
    # Lista de paquetes requeridos
    required_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "nltk>=3.7",
        "spacy>=3.4.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "textblob>=0.17.0",
        "wordcloud>=1.8.0",
        "tqdm>=4.64.0"
    ]

    print("=== CONFIGURACIÓN DEL AMBIENTE ===")
    print("Instalando paquetes necesarios...")

    # Instalar paquetes
    for package in tqdm(required_packages, desc="Instalando paquetes"):
        install_package(package)

    print("\n=== DESCARGANDO MODELOS DE NLP ===")

    # Descargar datos de NLTK
    try:
        import nltk
        nltk_data = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet', 'omw-1.4']
        for data in tqdm(nltk_data, desc="Descargando datos NLTK"):
            nltk.download(data, quiet=True)
        print("✓ Datos NLTK descargados")
    except Exception as e:
        print(f"Error descargando NLTK: {e}")

    # Descargar modelo de spaCy para inglés
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ Modelo spaCy en_core_web_sm descargado")
    except Exception as e:
        print(f"Error descargando spaCy: {e}")

    return True

def download_transformer_models():
    """Descarga y verifica modelos transformer"""
    print("=== DESCARGANDO MODELOS TRANSFORMER ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Lista de modelos a precargar
        transformer_models = [
            "distilbert-base-uncased",
            "roberta-base", 
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
        
        downloaded_models = {}
        
        for model_name in transformer_models:
            try:
                print(f"Precargando {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                downloaded_models[model_name] = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'status': 'success'
                }
                
                print(f"✓ {model_name} descargado y cacheado")
            except Exception as e:
                print(f"⚠ Error con {model_name}: {e}")
                downloaded_models[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        print("✓ Modelos transformer listos")
        return downloaded_models
        
    except Exception as e:
        print(f"⚠ Error configurando transformers: {e}")
        return {}

def check_device():
    """Verifica disponibilidad de GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU disponible: {device_name}")
            return "cuda"
        else:
            print("✓ Usando CPU (GPU no disponible)")
            return "cpu"
    except Exception:
        print("✓ Usando CPU")
        return "cpu"

def get_environment_info():
    """Obtiene información del ambiente configurado"""
    try:
        import pandas as pd
        import sklearn
        import transformers
        import torch
        
        info = {
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": check_device()
        }
        
        print("=== INFORMACIÓN DEL AMBIENTE ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        return info
        
    except Exception as e:
        print(f"Error obteniendo información: {e}")
        return {}
