from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data_loader import SarcasmDataset
#from src.model import RCNN_BETO
from models.bert_tuit import RCNN_BERTUIT
from src.evaluate import evaluate
from src.utils import predecir
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Usando dispositivo:", device)

# 📥 Dataset y split reproducible
dataset = load_dataset("Ernesto-1997/Sarcastic_spanish_dataset")['train']
dataset = dataset.train_test_split(test_size=0.2, seed=42)
val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
dataset['test'] = val_test['test']

# 🔤 Tokenizador
tokenizer = AutoTokenizer.from_pretrained("dpysentimiento/robertuito-irony") # Puedes cambiar a otro tokenizador si lo deseas

# 📦 Dataset para test
test_dataset = SarcasmDataset(dataset['test'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)

# 🧠 Modelo
model = RCNN_BERTUIT()
model.load_state_dict(torch.load("modelo_sarcasmo.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Modelo cargado desde 'modelo_sarcasmo.pth'")

# 📊 Evaluación
evaluate(model, test_loader, device)

# 🔮 Prueba de predicción
texto_nuevo = "Oh genial, otro lunes por la mañana."
predecir(texto_nuevo, tokenizer, model, device)
