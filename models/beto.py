from transformers import AutoModelForSequenceClassification

def get_simple_beto_model():
    return AutoModelForSequenceClassification.from_pretrained(
        "dccuchile/bert-base-spanish-wwm-uncased",
        num_labels=1  # Salida binaria con BCEWithLogits
    )
