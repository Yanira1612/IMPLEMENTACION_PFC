from datasets import load_dataset
from collections import Counter

# Carga el dataset original
dataset = load_dataset("Ernesto-1997/Sarcastic_spanish_dataset")['train']

# División inicial como en tu main
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Accede al conjunto de entrenamiento
train_data = dataset['train']

# Cuenta cuántos ejemplos hay por clase
conteo = Counter([ej['Sarcasmo'] for ej in train_data])

# Muestra los resultados
print("Distribución de clases en el set de entrenamiento:")
print(f"No Sarcasmo (0): {conteo[0]}")
print(f"Sarcasmo     (1): {conteo[1]}")
