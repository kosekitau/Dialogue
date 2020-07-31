# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

import torchtext
from torchtext.data.utils import get_tokenizer

MAX_LENGTH = 30

#テキストに処理を行うFieldを定義
#fix_lengthはtokenの数
SRC = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenizer_with_preprocessing,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH)

TRG = torchtext.data.Field(sequential=True, use_vocab=True, tokenize=tokenizer_with_preprocessing,
                            lower=True, include_lengths=True, batch_first=True, fix_length=MAX_LENGTH,
                            init_token='<cls>', eos_token='<eos>')

LABEL_SRC = torchtext.data.Field(sequential=False, use_vocab=False)

LABEL_TRG = torchtext.data.Field(sequential=False, use_vocab=False)

#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/dataset/DailyDialog/', train='train.csv', validation='validation.csv',
    test='test.csv', format='csv', fields=[('src', SRC), ('trg', TRG), ('label_src', LABEL_SRC), ('label_trg', LABEL_TRG)])

SRC.build_vocab(train_ds)
TRG.build_vocab(train_ds)
print(TRG.vocab.stoi)
print(len(TRG.vocab.stoi))

from torchtext import data

train_dl = data.Iterator(train_ds, batch_size=64, train=True)
val_dl = data.Iterator(val_ds, batch_size=64, train=False, sort=False)
batch = next(iter(val_dl))
print(batch.src[0].shape)
print(batch.trg[0].shape)
print(batch.label_src.shape)

class EncoderRNN(nn.Module):
  def __init__(self, hidden_size, vocab_size, emotion_size, dropout=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
    self.thought = nn.Linear(hidden_size*2+emotion_size, hidden_size)


  def forward(self, input_seq, emotion, hidden=None):
    embedded = self.embedding(input_seq) #[64, 30, 600]
    outputs, (hn, cn) = self.lstm(embedded) #[64, 30, 1200], ([2, 64, 600], [2, 64, 600])
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] #[64, 30, 600]
    thought_vector = torch.cat((hn[0], hn[1], emotion), -1) #[64, 1200]
    thought_vector = self.thought(thought_vector).unsqueeze(0) #[1, 64, 600]

    return outputs, thought_vector

def label2one_hot(emotion):
  result = torch.zeros(batch_size, emotion_size)
  for i, e in enumerate(emotion):
    result[i][e] = 1.
  return result.to(device)

print(device)
#エンコーダテスト
hidden_size = 600
dropout = 0.1
batch_size = 64
emotion_size = 7

batch = next(iter(val_dl))
emotion = batch.label_src
emotion = label2one_hot(emotion)
encoder.to(device)

encoder = EncoderRNN(hidden_size, len(SRC.vocab.stoi), emotion_size, dropout)
encoder_outputs, thought_vector = encoder(batch.src[0].to(device), emotion.to(device))

class LuongAttnDecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size, dropout=0.1):
    super(LuongAttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout = dropout

    self.embedding = nn.Embedding(output_size, hidden_size)
    self.embedding_dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
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

"""
decoder = LuongAttnDecoderRNN(hidden_size, len(TRG.vocab.stoi),  dropout)
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
"""

def a_batch_loss(input_variable, target_variable, emotion, max_target_len, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion, phase):
  total_loss = 0 #1batchのloss
  # Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  n_totals = 0
  print_losses = []

  #エンコーダの出力
  encoder_outputs, thought_vector = encoder(input_variable, emotion)
  #['<cls>']を生成
  decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>'] for _ in range(batch_size)]) #[64]
  decoder_input = decoder_input.to(device)
  #エンコーダの最後の隠れ状態を使用、記憶セルは0を入力
  cn = torch.zeros(1, batch_size, hidden_size, device=device)
  decoder_hidden = (thought_vector, cn)

  #teaching_forceを使う
  loss = 0 #1batchの中の1センテンスのloss
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      # Teacher forcing: next input is current target
      decoder_input = target_variable[:, t] #[64], teaching_forceの場合、正解データを次に入力する
      #loss += criterion(decoder_output, target_variable[:, t])
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
    #total_loss += loss / max_target_len #1バッチ分のloss

  else:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      # Teacher forcing: next input is current target
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

def train_model(dataloaders_dict, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
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
        target_variable = batch.trg[0].to(device) #(64, 30)
        emotion = batch.label_src
        emotion = label2one_hot(emotion)
        max_target_len = max(batch.trg[1])
        if target_variable.shape[0] == 64:
          total_loss = a_batch_loss(input_variable, target_variable, emotion, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, phase) #1バッチ分のloss
          print_loss += total_loss #1epochのlossをprint_lossに加えていく

      #損失をだす
      print("epoch: {}; phase: {}; Average loss: {:.4f}; PPL: {:.4f}".format(epoch+1, phase, print_loss/i, math.exp(print_loss/i) ))

hidden_size = 600
dropout = 0.1
batch_size = 64
emotion_size = 7

encoder = EncoderRNN(hidden_size, len(SRC.vocab.stoi), emotion_size, dropout)
decoder = LuongAttnDecoderRNN(hidden_size, len(TRG.vocab.stoi),  dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

from torch import optim

clip = 1.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_rate = 5.0
num_epochs = 5

dataloaders_dict = {"train": train_dl, "val": val_dl}

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_rate)

encoder.train()
decoder.train()

train_model(dataloaders_dict, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

import pickle

model_path = '/content/drive/My Drive/dataset/DailyDialog/chatbot_encoder0725.pth'
torch.save(encoder.to('cpu').state_dict(), model_path)
model_path = '/content/drive/My Drive/dataset/DailyDialog/chatbot_decoder0725.pth'
torch.save(decoder.to('cpu').state_dict(), model_path)

model_path = '/content/drive/My Drive/dataset/DailyDialog/cuda_chatbot_encoder0725.pth'
torch.save(encoder.to('cuda').state_dict(), model_path)
model_path = '/content/drive/My Drive/dataset/DailyDialog/cuda_chatbot_decoder0725.pth'
torch.save(decoder.to('cuda').state_dict(), model_path)

with open('/content/drive/My Drive/dataset/DailyDialog/voc_word2index0725.pkl', 'wb') as f:
    pickle.dump(SRC.vocab.stoi, f)
with open('/content/drive/My Drive/dataset/DailyDialog/voc_index2word0725.pkl', 'wb') as f:
    pickle.dump(TRG.vocab.itos, f)

hidden_size = 600
dropout = 0.1
batch_size = 64
emotion_size = 7

encoder = EncoderRNN(hidden_size, len(SRC.vocab.stoi), emotion_size, dropout)
decoder = LuongAttnDecoderRNN(hidden_size, len(TRG.vocab.stoi),  dropout)
encoder.load_state_dict(torch.load('/content/drive/My Drive/dataset/DailyDialog/cuda_chatbot_encoder0725.pth'))
decoder.load_state_dict(torch.load('/content/drive/My Drive/dataset/DailyDialog/cuda_chatbot_decoder0725.pth'))
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, thought_vector = self.encoder(input_seq)
        cn = torch.zeros(1, 1, hidden_size).to(device)
        decoder_hidden = (thought_vector, cn)
        decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>']]).to(device)
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            #decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores

import unicodedata

def indexesFromSentence(sentence):
    return [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sentence.split(' ')]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def evaluate(encoder, decoder, searcher, sentence, max_length=MAX_LENGTH):
    indexes_batch = indexesFromSentence(sentence)
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
            #ノイズ処理
            input_sentence = normalizeString(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, input_sentence)
            output_words[:] = [x for x in output_words if not (x == '<eos>' or x == '<pad>')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

import copy
from heapq import heappush, heappop

class BeamSearchNode(object):
    def __init__(self, h, prev_node, wid, logp, length):
        self.h = h
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)

def beam_search_decoding(decoder, encoder_output, thought_vector, beam_width, n_best, sos_token, eos_token, max_dec_steps, device):
    assert beam_width >= n_best

    n_best_list = []
    cn = torch.zeros(1, 1, 600).to(device)
    decoder_hidden = (thought_vector, cn) #((1,1,600), (1,1,600))
    decoder_input = torch.LongTensor([sos_token]).to(device) #[1]
    end_nodes = []
    node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)
    nodes = []
    heappush(nodes, (-node.eval(), id(node), node))
    n_dec_steps = 0

    # Start beam search
    while True:
        #最大単語数を越したら
        if n_dec_steps > max_dec_steps:
            break

        # Fetch the best node
        score, _, n = heappop(nodes)
        decoder_input = n.wid #(1)
        decoder_hidden = n.h  #((1,1,600), (1,1,600))

        if n.wid.item() == eos_token and n.prev_node is not None:
            end_nodes.append((score, id(n), n))
            # If we reached maximum # of sentences required
            if len(end_nodes) >= n_best:
                break
            else:
                continue

         #(1, 単語数), ((1,1,600), (1,1,600))
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

        topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width) # (1, bw), (1, bw)


        for new_k in range(beam_width):
            decoded_t = topk_indexes[0][new_k].view(1) # (1) new_k番目のindexを取り出す
            logp = topk_log_prob[0][new_k].item() # new_k番目の単語の生成確率を取り出す
            node = BeamSearchNode(h=decoder_hidden,
                                  prev_node=n,
                                  wid=decoded_t,
                                  logp=n.logp+logp,
                                  length=n.length+1)
            heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps += beam_width

    #ここでnodesに追加する作業が終わる
    # if there are no end_nodes, retrieve best nodes (they are probably truncated)
    if len(end_nodes) == 0:
        end_nodes = [heappop(nodes) for _ in range(beam_width)]

    # Construct sequences from end_nodes
    n_best_seq_list = []
    for score, _id, n in sorted(end_nodes, key=lambda x: x[0]):
        sequence = [n.wid.item()]
        # back trace from end node
        while n.prev_node is not None:
            n = n.prev_node
            sequence.append(n.wid.item())
        sequence = sequence[::-1] # reverse

        n_best_seq_list.append(sequence)

    n_best_list.append(n_best_seq_list)

    return n_best_list

def print_n_best(decoded_seq, itos):
    for rank, seq in enumerate(decoded_seq):
        print(f'Out: Rank-{rank+1}: {" ".join([itos[idx] for idx in seq])}')

encoder.eval()
decoder.eval()

beam_width = 10
n_best = 5

def label2one_hot(emotion):
  result = torch.zeros(1, emotion_size)
  for i, e in enumerate(emotion):
    result[i][e] = 1.
  return result.to(device)

src = normalizeString('Oh, I’m sorry I bothered you. I’m really sorry.').lower()
indexes_batch = indexesFromSentence(src)
input_batch = torch.LongTensor(indexes_batch).view(1, -1).to(device)
emotion = label2one_hot([5])
encoder_outputs, thought_vector = encoder(input_batch, emotion)
n_best_list = beam_search_decoding(decoder, encoder_outputs, thought_vector, beam_width, n_best, TRG.vocab.stoi['<cls>'], TRG.vocab.stoi['<eos>'],  1000, device='cuda')
print_n_best(n_best_list[0], TRG.vocab.itos)
