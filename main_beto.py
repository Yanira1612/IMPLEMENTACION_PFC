from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data_loader import SarcasmDataset
from models.beto import get_simple_beto_model
from src.train import train_model
from src.evaluate import evaluate
from src.utils import predecir
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Usando dispositivo:", device)

# ⚙️ Cambiar entre modelos
USE_RCNN = False

# 📥 Dataset
dataset = load_dataset("Ernesto-1997/Sarcastic_spanish_dataset")['train']
dataset = dataset.train_test_split(test_size=0.2, seed=42)
val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
dataset['val'] = val_test['train']
dataset['test'] = val_test['test']

# 🔤 Tokenizador (ambos modelos usan el mismo tokenizer por ahora)
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")

# 📦 Dataset a tensores
train_dataset = SarcasmDataset(dataset['train'], tokenizer)
val_dataset = SarcasmDataset(dataset['val'], tokenizer)
test_dataset = SarcasmDataset(dataset['test'], tokenizer)

# 🧪 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# 🧠 Modelo

model = get_simple_beto_model()
model_name = "modelo_simple_bertuit.pth"

# 🚂 Entrenar
train_model(model, train_dataset, val_dataset, tokenizer, device=device)

# 💾 Guardar modelo
torch.save(model.state_dict(), model_name)
print(f"\n💾 Modelo guardado como '{model_name}'")

# 📊 Evaluación
evaluate(model, test_loader, device)

# 🔮 Predicción
texto_nuevo = "Oh genial, otro lunes por la mañana."
predecir(texto_nuevo, tokenizer, model, device)
