# -*- coding: utf-8 -*-
"""TIU_Bi_2layer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uu5TugKUsFZ5UmD1rcoCMhxrAzqF1uck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class EncoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, num_layers, bidirectional, vocab_size, text_embedding_vectors, dropout=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    if text_embedding_vectors == None:
      self.embedding = nn.Embedding(vocab_size, emb_size)
    else:
      self.embedding = nn.Embedding.from_pretrained(
          embeddings=text_embedding_vectors, freeze=True)
    self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
    self.thought = nn.Linear(hidden_size*2, hidden_size)


  def forward(self, input_seq, hidden=None):
    embedded = self.embedding(input_seq) #[batch, max_length, emb_size]
    outputs, (hn, cn) = self.lstm(embedded) #[batch, max_length, hidden*2], ([2, 64, 600], [2, 64, 600])
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] #[batch, max_length, hidden]

    encoder_hidden = tuple([hn[i, :, :] + hn[i+1, :, :] for i in range(0, self.num_layers+2*self.bidirectional, 2)])
    encoder_hidden = torch.stack(encoder_hidden, 0)

    return outputs, encoder_hidden

en = EncoderRNN(300, 600, 2, True, 2000, None, dropout=0)
outputs, encoder_hidden = en(torch.randint(0, 1000, size=(128, 20)))

torch.stack(a, 0).shape

lstm = nn.LSTM(300, 600, 2, batch_first=True, bidirectional=True)
input = torch.randn(128, 20, 300)
h0 = torch.randn(4, 128, 600)
c0 = torch.randn(4, 128, 600)
output, (hn, cn) = lstm(input, (h0, c0))

output.shape

hn.shape

#num_layer, num_directions
hn2 = hn.view(2, 2, 128, 600)

hn[3, 0, :] == hn2[1, 1, 0, :]

hn[0]

output[0, 0, :600].shape

hn[1, 0, 0, :].shape

output[0, 0, 600:] == hn[1, 1, 0, :]

2*False

