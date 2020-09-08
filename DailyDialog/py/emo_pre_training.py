# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#学習済みの分散表現をロードする
from torchtext.vocab import Vectors

english_fasttext_vectors = Vectors(name='drive/My Drive/wiki-news-300d-1M.vec')

print(english_fasttext_vectors.dim)
print(len(english_fasttext_vectors.itos))

import string
import re

# 以下の記号はスペースに置き換えます（カンマ、ピリオドを除く）。
# punctuationとは日本語で句点という意味です
print("区切り文字：", string.punctuation)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# 前処理
def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)
    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")
    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text

# 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）
def tokenizer_punctuation(text):
    return text.strip().split()

# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


# 動作を確認します
print(tokenizer_with_preprocessing('I like cats+'))

import pandas as pd

df = pd.read_csv('/content/drive/My Drive/dataset/Cornell-Movie-Quotes-Corpus/val.csv')
df.head()

import torchtext
from torchtext.data.utils import get_tokenizer

MAX_LENGTH = 30

#テキストに処理を行うFieldを定義
#fix_lengthはtokenの数
SRC = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenizer_with_preprocessing,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            eos_token='<eos>')

TRG = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenizer_with_preprocessing,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            init_token='<cls>', eos_token='<eos>')

LABEL_SRC = torchtext.data.Field(sequential=False, use_vocab=False)

LABEL_TRG = torchtext.data.Field(sequential=False, use_vocab=False)

"""
#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/dataset/DailyDialog/', train='train.csv', validation='validation.csv',
    test='test.csv', format='csv', fields=[('src', SRC), ('trg', TRG), ('label_src', LABEL_SRC), ('label_trg', LABEL_TRG)])
"""


#映画コーパス#/content/drive/My Drive/dataset/Cornell-Movie-Quotes-Corpus/
train_ds, val_ds = torchtext.data.TabularDataset.splits(
    path='/content/drive/My Drive/dataset/Cornell-Movie-Quotes-Corpus/', train='train.csv', validation='val.csv',
    format='csv', fields=[('src', SRC), ('trg', TRG)])

SRC.build_vocab(train_ds, vectors=english_fasttext_vectors)
TRG.build_vocab(train_ds, vectors=english_fasttext_vectors)
#SRC.build_vocab(train_ds)
#TRG.build_vocab(train_ds)
print(TRG.vocab.stoi)
print(len(TRG.vocab.stoi))

from torchtext import data

batch_size = 64

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
    self.embedding_dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)


  def forward(self, input_seq, hidden=None):
    embedded = self.embedding(input_seq) #[batch, sentence, hidden]
    embedded = self.embedding_dropout(embedded)
    outputs, (hn, cn) = self.lstm(embedded) #[batch, sentence, hidden*2], ([2, batch, hidden], [2, batch, hidden])
    #エンコーダの隠れ状態前後ろを足す
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] #[64, 30, 600]
    #attention用の隠れ状態とhn
    return outputs, hn

class Emotion_Encoder(nn.Module):
  def __init__(self, encoder, hidden_size, emotion_size):
    super(Emotion_Encoder, self).__init__()
    self.encoder = encoder
    self.fully_connected = nn.Linear(hidden_size*2+emotion_size, hidden_size)

  def forward(self, input_seq, hidden=None):
    encoder_output, hn = self.encoder(input_seq, hidden)
    thought_vector = torch.cat((hn[0], hn[1]), -1) #[64, 1200]
    thought_vector = self.fully_connected(thought_vector).unsqueeze(0) #[1, 64, 600]
    return encoder_outputs, thought_vector

def label2one_hot(emotion):
  result = torch.zeros(batch_size, emotion_size)
  for i, e in enumerate(emotion):
    result[i][e] = 1.
  return result.to(device)

class AttnDecoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, text_embedding_vectors, output_size, dropout=0.1):
    super(AttnDecoderRNN, self).__init__()
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
    self.concat_dropout = nn.Dropout(dropout)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input_step, decoder_hidden, encoder_outputs):
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    embedded = embedded.unsqueeze(1) #[64, 1, 600]

    #記憶セルはencoderから引っ張ってこない
    print(embedded.shape, decoder_hidden[0].shape, decoder_hidden[1].shape)
    rnn_output, hidden = self.lstm(embedded, decoder_hidden) #[64, 1, 600] ([1, 64, 600], [1, 64, 600])
    energy = self.score(encoder_outputs) # [64, 30, 600]
    attn_weights = torch.sum(rnn_output*energy, dim=2) #[64, 30]
    attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1) # [64, 1, 30]

    context = attn_weights.bmm(encoder_outputs) #[64, 1, 600]
    rnn_output = rnn_output.squeeze(1) #[64, 600]
    context = context.squeeze(1) #[64, 600]
    concat_input = torch.cat((rnn_output, context), 1) #[64, 1200]
    concat_input = self.concat_dropout(concat_input)
    concat_output = torch.tanh(self.concat(concat_input))
    output = self.out(concat_output)

    return output, hidden

en = EncoderRNN(300, 600, len(TRG.vocab.stoi), None)
en_e = Emotion_Encoder(en, 600, 0)
encoder_outputs, hn = en_e(batch.src[0])

de = AttnDecoderRNN(300, 600, None, len(TRG.vocab.stoi))

cn = torch.zeros(1, batch_size, 600)
decoder_hidden = (hn, cn)
decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>'] for _ in range(batch_size)]) #[64]
decoder_output, decoder_hidden = de(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
