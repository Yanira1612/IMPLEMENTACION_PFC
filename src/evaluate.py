import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Evaluando"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.squeeze()  # <- aquí adaptamos si logits es [batch_size, 1]

            probs = torch.sigmoid(logits)  # <- usamos sigmoid para salida binaria
            preds = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 🎯 Métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = "AUC no disponible (una sola clase)"

    print("\n📊 Resultados del modelo en el conjunto de evaluación:")
    print(f"🔹 Accuracy : {accuracy:.4f}")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall   : {recall:.4f}")
    print(f"🔹 F1-score : {f1:.4f}")
    print(f"🔹 AUC      : {auc}")