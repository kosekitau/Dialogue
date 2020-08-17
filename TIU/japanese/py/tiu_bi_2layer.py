# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import torchtext
from torchtext.data.utils import get_tokenizer

MAX_LENGTH = 20

#テキストに処理を行うFieldを定義
#fix_lengthはtokenの数
SRC = torchtext.data.Field(sequential=True, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            eos_token='<eos>')

TRG = torchtext.data.Field(sequential=True, use_vocab=True,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            init_token='<cls>', eos_token='<eos>')

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds = torchtext.data.TabularDataset.splits(
    path='/content/drive/My Drive/dataset/TIU/twitter/5_20', train='train.csv', validation='val.csv',
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

class DecoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, num_layers, text_embedding_vectors, output_size, dropout=0.1):
    super(DecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout = dropout
    if text_embedding_vectors == None:
      self.embedding = nn.Embedding(output_size, emb_size)
    else:
      self.embedding = nn.Embedding.from_pretrained(
          embeddings=text_embedding_vectors, freeze=True)
    self.embedding_dropout = nn.Dropout(self.dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
    self.out_dropout = nn.Dropout(self.dropout)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input_step, decoder_hidden, encoder_outputs):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    embedded = embedded.unsqueeze(1) #[batch, 1, hidden]

    #記憶セルはencoderから引っ張ってこない
    rnn_output, hidden = self.lstm(embedded, decoder_hidden) #[128, 1, 600] ([1, batch, hidden], [1, batch, hidden])
    attn_weights = torch.matmul(rnn_output, encoder_outputs.transpose(2, 1))
    attn_weights = F.softmax(attn_weights, -1)
    attn_applied = torch.bmm(attn_weights, encoder_outputs)
    output = rnn_output + attn_applied
    output = output.squeeze(1)
    output = self.out_dropout(output)
    output = self.out(output)
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

def a_batch_loss(input_variable, target_variable, max_target_len, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, phase):
  total_loss = 0 #1batchのloss
  # Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  n_totals = 0
  print_losses = []

  #エンコーダの出力
  encoder_outputs, thought_vector = encoder(input_variable)
  #['<cls>']を生成
  decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>'] for _ in range(batch_size)]) #[64]
  decoder_input = decoder_input.to(device)
  #エンコーダの最後の隠れ状態を使用、記憶セルは0を入力
  cn = torch.zeros(3, batch_size, hidden_size, device=device)
  decoder_hidden = (thought_vector, cn)

  #teaching_forceを使う
  loss = 0 #1batchの中の1センテンスのloss
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    for t in range(max_target_len - 1):
      decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      decoder_input = target_variable[:, t] #[64], teaching_forceの場合、正解データを次に入力する
      #loss += criterion(decoder_output, target_variable[:, t])
      #各バッチのtのlossをだす。mask_lossはnTotalで割った平均、nTotalはバッチ数からmask(<pad>)の数を引いたもの
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
    #total_loss += loss / max_target_len #1バッチ分のloss

  else:
    for t in range(max_target_len - 1):
      decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([topi[i] for i in range(batch_size)])
      decoder_input = decoder_input.to(device)
      #loss += criterion(decoder_output, target_variable[:, t])
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
    #total_loss += (loss / max_target_len) #1バッチ分のloss

  if phase == 'train':
    loss.backward()
    #total_loss.backward()
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()
  return sum(print_losses) / n_totals
  #return total_loss #1バッチ分のloss

import random

def train_model(dataloaders_dict, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer):
  print("Training...")
  #エポック
  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        encoder.train()
        decoder.train()
      else:
        encoder.eval()
        decoder.eval()
      print_loss = 0 #1epochのloss

      for i, batch in enumerate(dataloaders_dict[phase]):
        input_variable = batch.src[0].to(device) #(64, 30)
        target_variable = batch.trg[0][:, 1:].to(device) #(64, 30)
        max_target_len = max(batch.trg[1])
        if target_variable.shape[0] == batch_size:
          total_loss = a_batch_loss(input_variable, target_variable, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, phase) #1バッチ分のloss
          print_loss += total_loss #1epochのlossをprint_lossに加えていく

      #損失をだす
      print("epoch: {}; phase: {}; Average loss: {:.4f}; PPL: {:.4f}".format(epoch+1, phase, print_loss/i, math.exp(print_loss/i) ))

emb_size = 300
hidden_size = 700
num_layers = 2
bidirectional = True
dropout = 0.7

clip = 1.0
teacher_forcing_ratio = 1.0
learning_rate = 0.002
decoder_learning_rate = 1.0
num_epochs = 10

encoder = EncoderRNN(emb_size, hidden_size, num_layers, bidirectional, len(SRC.vocab.stoi), SRC.vocab.vectors, dropout)
decoder = DecoderRNN(emb_size, hidden_size, num_layers, TRG.vocab.vectors, len(TRG.vocab.stoi),  dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)

from torch import optim

dataloaders_dict = {"train": train_dl, "val": val_dl}

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_rate )

encoder.train()
decoder.train()


train_model(dataloaders_dict, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer)
