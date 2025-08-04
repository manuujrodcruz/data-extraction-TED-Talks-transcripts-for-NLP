"""
Módulo para mostrar progreso en tiempo real
"""

import time
import sys
from datetime import datetime

class ProgressTracker:
    """Clase para mostrar progreso en tiempo real"""
    
    def __init__(self, total_steps=None, description="Procesando"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        
    def start(self, message=None):
        """Inicia el seguimiento de progreso"""
        if message:
            print(f"Iniciando: {message}")
        print(f"Tiempo de inicio: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        self.start_time = time.time()
        
    def step(self, message=None, details=None):
        """Avanza un paso en el progreso"""
        self.current_step += 1
        current_time = time.time()
        step_duration = current_time - self.start_time
        self.step_times.append(step_duration)
        
        if self.total_steps:
            progress = (self.current_step / self.total_steps) * 100
            print(f"[{self.current_step}/{self.total_steps}] ({progress:.1f}%) ", end='')
        else:
            print(f"[{self.current_step}] ", end='')
            
        if message:
            print(f"{message}...", end='', flush=True)
            
        # Esperar un momento para mostrar el progreso
        time.sleep(0.1)
        
        if details:
            print(f" - {details}")
        else:
            print(" OK")
            
    def finish(self, message="Completado"):
        """Termina el seguimiento y muestra resumen"""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 50)
        print(f"Estado: {message}")
        print(f"Tiempo total: {total_time:.1f} segundos")
        print(f"Finalizado: {datetime.now().strftime('%H:%M:%S')}")
        
        if self.total_steps:
            avg_time = total_time / self.total_steps
            print(f"Tiempo promedio por paso: {avg_time:.1f}s")
            
def show_progress(func):
    """Decorador para mostrar progreso automáticamente"""
    def wrapper(*args, **kwargs):
        print(f"Ejecutando {func.__name__}...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} completado en {end-start:.1f}s")
        return result
    return wrapper

def print_step(step_num, total_steps, message, status="Procesando"):
    """Imprime un paso individual con formato consistente"""
    progress = (step_num / total_steps) * 100
    print(f"{status} [{step_num}/{total_steps}] ({progress:.1f}%) {message}")

def print_substep(message, status="  ->"):
    """Imprime un sub-paso con indentación"""
    print(f"{status} {message}")

def real_time_feedback(message):
    """Muestra feedback en tiempo real con timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# Ejemplos de uso:
def example_usage():
    """Ejemplo de cómo usar el ProgressTracker"""
    
    # Ejemplo 1: Con número total conocido
    tracker = ProgressTracker(total_steps=5, description="Configurando ambiente")
    tracker.start("Iniciando configuración del proyecto")
    
    tracker.step("Instalando pandas", "Biblioteca para manejo de datos")
    time.sleep(1)  # Simular trabajo
    
    tracker.step("Instalando scikit-learn", "Biblioteca de machine learning")
    time.sleep(1)
    
    tracker.step("Configurando NLTK")
    time.sleep(1)
    
    tracker.step("Verificando instalaciones")
    time.sleep(1)
    
    tracker.step("Creando configuración final")
    time.sleep(1)
    
    tracker.finish("Configuración del ambiente completada")
    
    print("\n" + "="*50)
    
    # Ejemplo 2: Sin número total (proceso dinámico)
    tracker2 = ProgressTracker(description="Procesando datos")
    tracker2.start("Iniciando procesamiento de datos")
    
    for i in range(3):
        tracker2.step(f"Procesando lote {i+1}")
        time.sleep(0.5)
        
    tracker2.finish("Procesamiento de datos completado")

if __name__ == "__main__":
    example_usage()
