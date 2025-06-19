import torch
from collections import Counter
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

def train_model(model, train_dataset, val_dataset, tokenizer, batch_size=8, lr=1e-3, epochs=5, device='cpu'):
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ✅ Obtener los pesos de clase
    labels = [ej['labels'].item() if isinstance(ej['labels'], torch.Tensor) else ej['labels'] for ej in train_dataset]
    conteo = Counter(labels)
    print("📊 Distribución clases:", conteo)

    # ⚖️ Calcular pos_weight: más peso a la clase minoritaria
    pos_weight = torch.tensor([conteo[0] / conteo[1]]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"🧪 Entrenando Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} - Pérdida de entrenamiento: {total_loss / len(train_loader):.4f}")

        # Evaluar
        evaluate(model, val_loader, device)

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
          #  predictions = (outputs > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"🎯 Exactitud en validación: {acc:.4f}")  