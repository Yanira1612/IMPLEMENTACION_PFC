import torch

def predecir(texto, tokenizer, model, device):
    model.eval()

    # Tokenizar el texto
    inputs = tokenizer(texto, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)
        prob = torch.sigmoid(logits)  # ahora aplicamos sigmoid aquí manualmente
        pred = (prob > 0.5).int().item()

    etiqueta = "Sí" if pred == 1 else "No"
    print(f"\n🗣️ Texto: {texto}")
    print(f"🔮 ¿Sarcástico?: {etiqueta} (Prob: {prob.item():.2f})")

