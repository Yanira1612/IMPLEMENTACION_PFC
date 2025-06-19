import torch
import torch.nn as nn
from transformers import AutoModel

class RCNN_BERTUIT(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=1, dropout_prob=0.3):
        super(RCNN_BERTUIT, self).__init__()
        
        # Cargar el modelo BETO (BERT entrenado para español)  
        #self.bert = AutoModel.from_pretrained("pysentimiento/robertuito-irony")
        self.bert = AutoModel.from_pretrained("pysentimiento/robertuito-base-uncased")
        
        # Capa BiLSTM para capturar dependencias contextuales
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout_prob)
        
        # Capa final lineal
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 por bidireccionalidad
        
        # Activación sigmoid para clasificación binaria
    #    self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask,**kwargs):
        # Obtener embeddings de BERT (puedes quitar no_grad si deseas entrenarlo también)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # output shape: [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # Pasar a través de la BiLSTM
        lstm_out, _ = self.lstm(sequence_output)  # output shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Pooling: tomar el promedio a lo largo de la secuencia
        pooled = torch.mean(lstm_out, dim=1)  # shape: [batch_size, hidden_dim * 2]
        
        # Capa de clasificación
        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits.squeeze()
