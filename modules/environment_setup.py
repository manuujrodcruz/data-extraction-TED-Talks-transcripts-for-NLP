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
        print(f"OK - {package} ya está instalado")
        return True
    except ImportError:
        try:
            print(f"Instalando {package}...", end='', flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120  # Aumentar timeout a 120 segundos
            )
            if result.returncode == 0:
                print(" OK")
                return True
            else:
                print(f" ERROR: {result.stderr[:100]}...")
                return False
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
            return False
        except Exception as e:
            print(f" ERROR: {str(e)[:100]}...")
            return False

def install_package_quick(package):
    """Versión rápida de instalación con timeout reducido"""
    try:
        package_name = package.split('==')[0].split('[')[0]
        __import__(package_name)
        print(f"OK - {package} ya está instalado")
        return True
    except ImportError:
        try:
            print(f"Instalando {package}...", end='', flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=60  # Timeout más corto para opcionales
            )
            if result.returncode == 0:
                print(" OK")
                return True
            else:
                print(" ERROR")
                return False
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
            return False
        except Exception:
            print(" ERROR")
            return False

def setup_environment():
    """Configura el ambiente completo del proyecto"""
    
    print("=== CONFIGURACION DEL AMBIENTE ===")
    print("Tiempo estimado: 2-5 minutos")
    print()
    
    # Lista de paquetes ESENCIALES (reducida para velocidad)
    essential_packages = [
        "pandas>=1.3.0",
        "numpy>=2.0.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "nltk>=3.7",
        "textblob>=0.17.0",
        "tqdm>=4.64.0"
    ]
    
    # Lista de paquetes OPCIONALES (solo si hay tiempo)
    optional_packages = [
        "plotly>=5.0.0",
        "spacy>=3.4.0",
        "wordcloud>=1.8.0"
    ]

    print(f"PASO 1/3: Instalando {len(essential_packages)} paquetes esenciales...")
    success_count = 0
    
    # Instalar paquetes esenciales
    for i, package in enumerate(essential_packages, 1):
        print(f"  [{i}/{len(essential_packages)}] ", end='', flush=True)
        if install_package(package):
            success_count += 1
    
    print(f"\nPaquetes esenciales: {success_count}/{len(essential_packages)} instalados")

    print(f"\nPASO 2/3: Instalando {len(optional_packages)} paquetes opcionales...")
    optional_success = 0
    
    # Instalar paquetes opcionales (con timeouts más cortos)
    for i, package in enumerate(optional_packages, 1):
        print(f"  [{i}/{len(optional_packages)}] ", end='', flush=True)
        if install_package_quick(package):
            optional_success += 1
    
    print(f"\nPaquetes opcionales: {optional_success}/{len(optional_packages)} instalados")

    print("\nPASO 3/3: Configurando modelos de NLP...")

    # Descargar datos de NLTK (esenciales)
    try:
        import nltk
        print("  Descargando datos NLTK...", end='', flush=True)
        essential_nltk = ['punkt', 'stopwords', 'vader_lexicon']
        for data in essential_nltk:
            nltk.download(data, quiet=True)
        print(" OK")
    except Exception as e:
        print(f" ERROR: {str(e)[:50]}...")

    # Verificar spaCy (opcional)
    try:
        import spacy
        print("  Verificando spaCy...", end='', flush=True)
        spacy.load("en_core_web_sm")  # Solo verificar que carga
        print(" OK")
    except Exception:
        print(" No disponible (se usará NLTK)")

    print("\nCONFIGURACION COMPLETADA")
    print("=" * 40)
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
