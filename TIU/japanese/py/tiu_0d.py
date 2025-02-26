# -*- coding: utf-8 -*-
"""TIU_0d.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O5fZA7EFfoO01bL1OOxHNQufjAN04BN4
"""

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
    path='/content/drive/My Drive/dataset/TIU/twitter/3_20', train='train_0907.csv', validation='val_0907.csv', 
    format='csv', fields=[('src', SRC), ('trg', TRG)])

#学習済みの分散表現をロードする
from torchtext.vocab import Vectors

japanese_fasttext_vectors = Vectors(name='/content/drive/My Drive/embedding/japanese_fasatext/model.vec')

print(japanese_fasttext_vectors.dim)
print(len(japanese_fasttext_vectors.itos))

"""
#SentencePiece
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
    path='/content/drive/My Drive/dataset/sentencepiece', train='sp_train_pre.csv', validation='sp_val_pre.csv', 
    format='csv', fields=[('src', SRC), ('trg', TRG)])
"""

SRC.build_vocab(train_ds, vectors=japanese_fasttext_vectors)
TRG.build_vocab(train_ds, vectors=japanese_fasttext_vectors)
#SRC.build_vocab(train_ds)
#TRG.build_vocab(train_ds)
print(TRG.vocab.stoi)
print(len(TRG.vocab.stoi))

from torchtext import data

batch_size = 256

train_dl = data.Iterator(train_ds, batch_size=batch_size, train=True)
val_dl = data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
batch = next(iter(val_dl))
print(batch.src[0].shape)
print(batch.trg[0].shape)
print([TRG.vocab.itos[b] for b in batch.trg[0][2]])

"""
!pip install sentencepiece > /dev/null

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("/content/drive/My Drive/dataset/sentencepiece/sentencepiece_twitter.model")
"""

#sp.EncodeAsPieces('タグに反応フォロバありがとうございました。呼びタメ大歓迎ですのでなかよくしてくださいっ')

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

    encoder_hidden = tuple([hn[i, :, :] + hn[i+1, :, :] for i in range(0, 6, 2)])
    encoder_hidden = torch.stack(encoder_hidden, 0)

    return outputs, encoder_hidden

en = EncoderRNN(300, 300, 3, True, 2000, None, dropout=0)
outputs, encoder_hidden = en(torch.randint(0, 1000, size=(batch_size, 20)))

"""
class DecoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, text_embedding_vectors, output_size, dropout=0.1):
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
    self.lstm1 = nn.LSTM(emb_size, hidden_size, batch_first=True)
    self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    lstm_list = [self.lstm1, self.lstm2, self.lstm3]
    self.module_list = nn.ModuleList(lstm_list)
    self.out_dropout = nn.Dropout(self.dropout)
    self.out = nn.Linear(hidden_size, output_size)
    
  def forward(self, input_step, decoder_hidden, encoder_outputs):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    embedded = embedded.unsqueeze(1) #[batch, 1, hidden]
    lstm_input = embedded
    
    hiddens = []
    #記憶セルはencoderから引っ張ってこない
    for i, lstm in enumerate(self.module_list):
      rnn_output, hidden = lstm(lstm_input, decoder_hidden[i]) #[128, 1, 300] ([1, batch, hidden], [1, batch, hidden])
      rnn_output = rnn_output + lstm_input
      attn_weights = torch.matmul(rnn_output, encoder_outputs.transpose(2, 1))
      attn_weights = F.softmax(attn_weights, -1)
      attn_applied = torch.bmm(attn_weights, encoder_outputs)
      lstm_input = rnn_output + attn_applied
      #lstm_input = torch.cat((rnn_output, attn_applied), dim=2)
      hiddens.append(hidden)
    
    output = lstm_input
    output = output.squeeze(1)
    output = self.out_dropout(output)
    output = self.out(output)
    output = F.softmax(output, dim=1)

    return output, hiddens
"""

class DecoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, text_embedding_vectors, output_size, dropout=0.1):
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
    self.embedding_norm = nn.LayerNorm(hidden_size)
    
    self.lstm1 = nn.LSTM(emb_size, hidden_size, batch_first=True)
    self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    # LayerNormalization層
    # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
    self.norm1 = nn.LayerNorm(hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size)
    self.norm3 = nn.LayerNorm(hidden_size)
    self.lstm_dropout1 = nn.Dropout(self.dropout)
    self.lstm_dropout2 = nn.Dropout(self.dropout)
    self.lstm_dropout3 = nn.Dropout(self.dropout)
    lstm_list = [self.lstm1, self.lstm2, self.lstm3]
    norm_list = [self.norm1, self.norm2, self.norm3]
    drop_list = [self.lstm_dropout1, self.lstm_dropout2, self.lstm_dropout3]
    self.l_module_list = nn.ModuleList(lstm_list)
    self.n_module_list = nn.ModuleList(norm_list)
    self.d_module_list = nn.ModuleList(drop_list)

    self.out_dropout = nn.Dropout(self.dropout)
    self.out = nn.Linear(hidden_size, output_size)
    
  def forward(self, input_step, decoder_hidden, encoder_outputs):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    embedded = self.embedding_norm(embedded)
    embedded = embedded.unsqueeze(1) #[batch, 1, hidden]
    lstm_input = embedded
    
    hiddens = []
    #記憶セルはencoderから引っ張ってこない
    for i, (lstm, norm, dropout) in enumerate(zip(self.l_module_list, 
                                                  self.n_module_list, self.d_module_list,)):
      rnn_output, hidden = lstm(lstm_input, decoder_hidden[i]) #[128, 1, 300] ([1, batch, hidden], [1, batch, hidden])
      lstm_input = rnn_output + lstm_input
      lstm_input = dropout(lstm_input)
      lstm_input = norm(lstm_input)
      hiddens.append(hidden)
    
    attn_weights = torch.matmul(lstm_input, encoder_outputs.transpose(2, 1))
    attn_weights = F.softmax(attn_weights, -1)
    attn_applied = torch.bmm(attn_weights, encoder_outputs)
    output = lstm_input + attn_applied
    output = output.squeeze(1)
    output = self.out_dropout(output)
    output = self.out(output)
    #output = F.softmax(output, dim=1)

    return output, hiddens

decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>'] for _ in range(batch_size)])
de = DecoderRNN(300, 300, None, 2000, dropout=0)
cn = torch.zeros(3, batch_size, 300)
hidden_list = []
for h, c in zip(encoder_hidden, cn):
  hidden_list.append((h.unsqueeze(0), c.unsqueeze(0)))
print(encoder_hidden.shape)
print(hidden_list[0][0].shape)
decoder_outputs, hiddens = de(decoder_input, hidden_list, outputs)

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
  #decoder_hidden = (thought_vector, cn)
  hidden_list =  [(h.unsqueeze(0), c.unsqueeze(0)) for h, c in zip(thought_vector, cn)]

  #teaching_forceを使う
  loss = 0 #1batchの中の1センテンスのloss
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    for t in range(max_target_len - 1):
      decoder_output, hidden_list = decoder(
          decoder_input, hidden_list, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      decoder_output = F.softmax(decoder_output, dim=1)
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
      decoder_output, hidden_list = decoder(
          decoder_input, hidden_list, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      decoder_output = F.softmax(decoder_output, dim=1)
      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([topi[i] for i in range(batch_size)])
      decoder_input = decoder_input.to(device)
      #loss += criterion(decoder_output, target_variable[:, t])
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])
      loss += mask_loss * nTotal
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
    #total_loss += (loss / max_target_len) #1バッチ分のloss
    
  if phase == 'train':
    loss = loss / n_totals
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
hidden_size = 300
num_layers = 3
bidirectional = True
dropout = 0.2

clip = 1.0
teacher_forcing_ratio = 1.0
learning_rate = 0.002
decoder_learning_rate = 1.0
num_epochs = 5

encoder = EncoderRNN(emb_size, hidden_size, num_layers, bidirectional, len(SRC.vocab.stoi), SRC.vocab.vectors, dropout)
decoder = DecoderRNN(emb_size, hidden_size, TRG.vocab.vectors, len(TRG.vocab.stoi),  dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)

from torch import optim

dataloaders_dict = {"train": train_dl, "val": val_dl}

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_rate )

encoder.train()
decoder.train()

train_model(dataloaders_dict, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer)

!apt install aptitude swig
!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
!pip install mecab-python3
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a
#https://medium.com/@jiraffestaff/mecabrc-%E3%81%8C%E8%A6%8B%E3%81%A4%E3%81%8B%E3%82%89%E3%81%AA%E3%81%84%E3%81%A8%E3%81%84%E3%81%86%E3%82%A8%E3%83%A9%E3%83%BC-b3e278e9ed07
!pip install unidic-lite

!pip install sentencepiece > /dev/null

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("/content/drive/My Drive/dataset/sentencepiece/sentencepiece_twitter.model")

import unicodedata
import re
import MeCab
import subprocess

cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,
                           shell=True).communicate()[0]).decode('utf-8')
mecab = MeCab.Tagger("-d {0} -Owakati".format(path))

#単語分割、id振り

def indexesFromSentence(sentence):
    return [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in mecab.parse(sentence)[:-2].split()]
"""   
def indexesFromSentence(sentence):
    return [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sp.EncodeAsPieces(sentence)]
"""

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(text):
    s = unicodeToAscii(text.lower().strip())
    s = re.sub(r"([.!?])", r"\1", text)
    s = re.sub(r"\s+", r" ", s).strip()
    s = re.sub(r'[【】]', '', s)                  # 【】の除去
    s = re.sub(r'[（）()]', '', s)                # （）の除去
    s = re.sub(r'[［］\[\]]', '', s)              # ［］の除去
    s = re.sub(r'[\r]', '', s)
    s = re.sub(r'　', ' ', s)                    #全角空白の除去
    s = re.sub(r'[^a-zA-Zぁ-んァ-ン一-龥0-9、。,.!?ー ]', '', s)
    return s


def evaluate(encoder, decoder, searcher, sentence, max_length):
    print(sentence)
    indexes_batch = indexesFromSentence(sentence)
    print(indexes_batch)
    print([SRC.vocab.itos[i] for i in indexes_batch])
    lengths = len(indexes_batch)
    input_batch = torch.LongTensor(indexes_batch).view(1, -1).to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [TRG.vocab.itos[token.item()] for token in tokens]
    return decoded_words

def evaluate_beam(encoder, decoder, searcher, sentence, max_length=MAX_LENGTH):
    indexes_batch = indexesFromSentence(sentence) #[]
    lengths = len(indexes_batch)
    input_batch = torch.LongTensor(indexes_batch).view(1, -1).to(device) #[1, ]
    encoder_outputs, decoder_hidden = encoder(input_batch, label2one_hot([0])) #[64, 30, 600], [1, 64, 600]

    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [TRG.vocab.itos[token.item()] for token in tokens]
    return decoded_words  


def evaluateInput(encoder, decoder, searcher):
    input_sentence = ''
    while(1):
      try:
        input_sentence = input('> ').lower()
        if input_sentence == 'q' or input_sentence == 'quit': break
        #前処理
        input_sentence = normalizeString(input_sentence)
        output_words = evaluate(encoder, decoder, searcher, input_sentence, MAX_LENGTH)
        output_words[:] = [x for x in output_words if not (x == '<eos>' or x == '<pad>')]
        print('Bot:', ' '.join(output_words))

      except KeyError:
        print("Error: Encountered unknown word.")

import numpy as np

class GreedySearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedySearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    encoder_outputs, thought_vector = self.encoder(input_seq)
    cn = torch.zeros(3, 1, hidden_size).to(device)
    hidden_list =  [(h.unsqueeze(0), c.unsqueeze(0)) for h, c in zip(thought_vector, cn)]
    decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>']]).to(device)
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    for _ in range(max_length):
      decoder_output, hidden_list = self.decoder(decoder_input, hidden_list, encoder_outputs)
      decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
      all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
      all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            
    return all_tokens, all_scores

tempreture = 0.3

class GreedyTempreture(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedyTempreture, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    encoder_outputs, thought_vector = self.encoder(input_seq)
    cn = torch.zeros(3, 1, hidden_size).to(device)
    hidden_list =  [(h.unsqueeze(0), c.unsqueeze(0)) for h, c in zip(thought_vector, cn)]
    decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>']]).to(device)
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    for _ in range(max_length):
      decoder_output, hidden_list = self.decoder(decoder_input, hidden_list, encoder_outputs)#[batch, vocab]
      decoder_output = decoder_output.squeeze(0)
      decoder_output = torch.log(decoder_output) / tempreture
      decoder_output = torch.exp(decoder_output)
      decoder_output = decoder_output / sum(decoder_output)
      decoder_input = torch.tensor([torch.multinomial(decoder_output ,1)], device=device, dtype=torch.long)#decoder_output
      all_tokens = torch.cat((all_tokens, decoder_input))
      #all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            
    return all_tokens, _

lamb = 0.3
gamma = 5

class MMI(nn.Module):
  def __init__(self, encoder, decoder):
    super(MMI, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.lm_decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    #seq2seq用
    encoder_outputs, thought_vector = self.encoder(input_seq)
    cn = torch.zeros(3, 1, hidden_size).to(device)
    hidden_list =  [(h.unsqueeze(0), c.unsqueeze(0)) for h, c in zip(thought_vector, cn)]
    #言語モデル用
    lm_hidden_list = [(h.unsqueeze(0), c.unsqueeze(0)) for h, c in zip(torch.zeros(3, 1, hidden_size).to(device), cn)]
    lm_encoder_outputs = torch.zeros(1, MAX_LENGTH, hidden_size).to(device)
    #共通で使う
    decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>']]).to(device)
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    for i in range(max_length):
      #MMI
      if i <= gamma-1:
        #seq2seqの出力
        decoder_output, hidden_list = self.decoder(decoder_input, hidden_list, encoder_outputs)
        #言語モデルの出力
        lm_decoder_output, lm_hidden_list = self.lm_decoder(decoder_input, lm_hidden_list, lm_encoder_outputs)
        #decoder_output = F.softmax(decoder_output, dim=1)

        #decoder_input = torch.log(F.softmax(decoder_output - lamb*lm_decoder_output, dim=1))
        decoder_input = torch.log(F.softmax(decoder_output, dim=1) - F.softmax(lamb * lm_decoder_output, dim=1))
        #decoder_input = torch.log(decoder_output - lamb*lm_decoder_output)
        #decoder_input = torch.log(decoder_output) - lamb*torch.log(lm_decoder_output)
        decoder_scores, decoder_input = torch.max(decoder_input, dim=1)
        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
        all_scores = torch.cat((all_scores, decoder_scores), dim=0)
      else:
        decoder_output, hidden_list = self.decoder(decoder_input, hidden_list, encoder_outputs)
        decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
        all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            
    return all_tokens, all_scores

encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)
#searcher = GreedyTempreture(encoder, decoder)
#searcher = MMI(encoder, decoder)
evaluateInput(encoder, decoder, searcher)

"""
import pickle

model_path = '/content/drive/My Drive/model/TIU_encoder0903.pth'
torch.save(encoder.to('cpu').state_dict(), model_path)
model_path = '/content/drive/My Drive/model/TIU_decoder0903.pth'
torch.save(decoder.to('cpu').state_dict(), model_path)


model_path = '/content/drive/My Drive/model/cuda_TIU_encoder0903.pth'
torch.save(encoder.to('cuda').state_dict(), model_path)
model_path = '/content/drive/My Drive/model/cuda_TIU_decoder0903.pth'
torch.save(decoder.to('cuda').state_dict(), model_path)

with open('/content/drive/My Drive/model/src_word2index0903.pkl', 'wb') as f:
    pickle.dump(SRC.vocab.stoi, f)
with open('/content/drive/My Drive/model/trg_word2index0903.pkl', 'wb') as f:
    pickle.dump(TRG.vocab.itos, f)
"""

indexesFromSentence('今日も暑いですね')

sentence = '今日も暑いですね'
for word in mecab.parse(sentence):
  if word in SRC.vocab.stoi:
    print(word)
    print(SRC.vocab.stoi[word])
  else:
    print(word)
    print(SRC.vocab.stoi['<unk>'])

mecab.parse(sentence)[:-2].split()

