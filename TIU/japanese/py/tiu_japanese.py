# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import torchtext
from torchtext.data.utils import get_tokenizer

MAX_LENGTH = 40

#テキストに処理を行うFieldを定義
#fix_lengthはtokenの数
SRC = torchtext.data.Field(sequential=True, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            eos_token='<eos>')

TRG = torchtext.data.Field(sequential=True, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            eos_token='<eos>')

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds = torchtext.data.TabularDataset.splits(
    path='/content/drive/My Drive/dataset/TIU/twitter', train='train.csv', validation='val.csv',
    format='csv', fields=[('src', SRC), ('trg', TRG)])

#学習済みの分散表現をロードする
from torchtext.vocab import Vectors

japanese_fasttext_vectors = Vectors(name='/content/drive/My Drive/embedding/japanese_fasatext/model.vec')

print(japanese_fasttext_vectors.dim)
print(len(japanese_fasttext_vectors.itos))

SRC.build_vocab(train_ds, vectors=japanese_fasttext_vectors)
TRG.build_vocab(train_ds, vectors=japanese_fasttext_vectors)
#SRC.build_vocab(train_ds)
#TRG.build_vocab(train_ds)
print(TRG.vocab.stoi)
print(len(TRG.vocab.stoi))

from torchtext import data

batch_size = 128

train_dl = data.Iterator(train_ds, batch_size=batch_size, train=True)
val_dl = data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
batch = next(iter(val_dl))
print(batch.src[0].shape)
print(batch.trg[0].shape)

class EncoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, vocab_size, text_embedding_vectors, dropout=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    if text_embedding_vectors == None:
      self.embedding = nn.Embedding(vocab_size, emb_size)
    else:
      self.embedding = nn.Embedding.from_pretrained(
          embeddings=text_embedding_vectors, freeze=True)
    self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
    self.thought = nn.Linear(hidden_size*2, hidden_size)


  def forward(self, input_seq, hidden=None):
    embedded = self.embedding(input_seq) #[64, 30, 600]
    outputs, (hn, cn) = self.lstm(embedded) #[64, 30, 1200], ([2, 64, 600], [2, 64, 600])
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] #[64, 30, 600]
    thought_vector = torch.cat((hn[0], hn[1]), -1) #[64, 1200]
    thought_vector = self.thought(thought_vector).unsqueeze(0) #[1, 64, 600]

    return outputs, thought_vector

#エンコーダテスト
emb_size = 300
hidden_size = 600
dropout = 0.1

batch = next(iter(val_dl))
print(batch.src[0].shape)

encoder = EncoderRNN(emb_size, hidden_size, len(SRC.vocab.stoi), SRC.vocab.vectors, dropout)
encoder_outputs, thought_vector = encoder(batch.src[0])
print(thought_vector.shape)

class LuongAttnDecoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, text_embedding_vectors, output_size, dropout=0.1):
    super(LuongAttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout = dropout
    if text_embedding_vectors == None:
      self.embedding = nn.Embedding(output_size, emb_size)
    else:
      self.embedding = nn.Embedding.from_pretrained(
          embeddings=text_embedding_vectors, freeze=True)
    self.embedding_dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
    self.score = nn.Linear(hidden_size, hidden_size)
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input_step, decoder_hidden, encoder_outputs):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    embedded = embedded.unsqueeze(1) #[64, 1, 600]

    #記憶セルはencoderから引っ張ってこない
    rnn_output, hidden = self.lstm(embedded, decoder_hidden) #[64, 1, 600] ([1, 64, 600], [1, 64, 600])
    energy = self.score(encoder_outputs) # [64, 30, 600]
    attn_weights = torch.sum(rnn_output*energy, dim=2) #[64, 30]
    attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1) # [64, 1, 30]

    context = attn_weights.bmm(encoder_outputs) #[64, 1, 500]
    rnn_output = rnn_output.squeeze(1) #[64, 500]
    context = context.squeeze(1) #[64, 500]
    concat_input = torch.cat((rnn_output, context), 1) #[64, 1000]
    concat_output = torch.tanh(self.concat(concat_input))
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)

    return output, hidden

def binaryMatrix(l, value=TRG.vocab.stoi['<pad>']):
    m = []
    for i, seq in enumerate(l):
      if seq == TRG.vocab.stoi['<pad>']:
        m.append(False)
      else:
        m.append(True)
    return m

def maskNLLLoss(inp, target):
    mask = target
    mask = binaryMatrix(mask)
    mask = torch.BoolTensor(mask)
    mask = mask.to(device)
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

decoder = LuongAttnDecoderRNN(emb_size, hidden_size, TRG.vocab.vectors, len(TRG.vocab.stoi),  dropout)
decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>'] for _ in range(batch_size)])
embedding = nn.Embedding(len(SRC.vocab.stoi), hidden_size)
print(decoder_input.shape)

cn = torch.zeros(1, batch_size, hidden_size)
#cn = torch.zeros(1, batch_size, hidden_size, device=device)
decoder_hidden = (thought_vector, cn)
target_variable = batch.trg[0]


for t in range(30):
  decoder_output, decoder_hidden = decoder(
      decoder_input, decoder_hidden, encoder_outputs
  ) #[64, 単語種類数], [2, 64, 500]
  # Teacher forcing: next input is current target
  _, topi = decoder_output.topk(1)
  decoder_input = torch.LongTensor([topi[i] for i in range(batch_size)])
  mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])

decoder_output, decoder_hidden = decoder(
    decoder_input, decoder_hidden, encoder_outputs
) #[64, 単語種類数], [2, 64, 500]

print('decoder_output', decoder_output.shape)
print('decoder_hidden', decoder_hidden[0].shape)
