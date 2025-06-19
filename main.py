from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data_loader import SarcasmDataset
#from src.model import RCNN_BETO
from models.bert_tuit import RCNN_BERTUIT
from src.train import train_model
from src.evaluate import evaluate
from src.utils import predecir
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Usando dispositivo:", device)


# 📥 Dataset desde Hugging Face
dataset = load_dataset("Ernesto-1997/Sarcastic_spanish_dataset")['train']
dataset = dataset.train_test_split(test_size=0.2, seed=42)
val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
dataset['val'] = val_test['train']
dataset['test'] = val_test['test']

# 🔤 Tokenizador RCNN BETO
#tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

#tokenizer = AutoTokenizer.from_pretrained("bertin-project/bertin-roberta-base-spanish-sentiment") 

#SEGUNDO TOKENIZER
#tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-irony")

tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")

# Cargar datasets
train_dataset = SarcasmDataset(dataset['train'], tokenizer)
val_dataset = SarcasmDataset(dataset['val'], tokenizer)
test_dataset = SarcasmDataset(dataset['test'], tokenizer)


# 🧪 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)


# Modelo
model = RCNN_BERTUIT()

# Entrenar
train_model(model, train_dataset, val_dataset, tokenizer, device=device)

# Guardar modelo
torch.save(model.state_dict(), "modelo_sarcasmo_bertuit_base.pth")
print("\n💾 Modelo guardado como 'modelo_sarcasmo.pth'")


# 📊 Evaluación final
evaluate(model, test_loader, device)

# 📝 Predicción
texto_nuevo = "Oh genial, otro lunes por la mañana."
predecir(texto_nuevo, tokenizer, model, device)