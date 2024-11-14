import torch
import torch.nn as nn
import torchvision.models as models

class ClassifierRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type='lstm', dropout=0.0):
        super(ClassifierRNN, self).__init__()

        # Selección entre LSTM o GRU
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("El tipo de RNN debe ser 'lstm' o 'gru'.")

        # Capa totalmente conectada para la clasificación binaria
        self.fc = nn.Linear(hidden_size, 1)

        # Función sigmoide para la salida binaria
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        rnn_out, _ = self.rnn(x)  # Salida de la RNN: (batch_size, sequence_length, hidden_size)

        # Aplicar la capa totalmente conectada en cada elemento del buffer
        logits = self.fc(rnn_out)  # (batch_size, sequence_length, 1)

        # Aplicar la función sigmoide para la salida binaria
        predictions = self.sigmoid(logits)  # (batch_size, sequence_length, 1)

        return predictions