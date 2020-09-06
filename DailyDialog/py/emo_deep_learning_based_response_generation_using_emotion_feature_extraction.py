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


#pandasでcsvを保存するときに、labelをintでキャストしておかないとエラーでるから注意
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='drive/My Drive/dataset/DailyDialog/', train='train.csv', validation='validation.csv',
    test='test.csv', format='csv', fields=[('src', SRC), ('trg', TRG), ('label_src', LABEL_SRC), ('label_trg', LABEL_TRG)])
"""

#映画コーパス#/content/drive/My Drive/dataset/Cornell-Movie-Quotes-Corpus/
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='/content/drive/My Drive/dataset/Cornell-Movie-Quotes-Corpus/', train='train.csv', validation='validation.csv',
    test='test.csv', format='csv', fields=[('src', SRC), ('trg', TRG), ('label_src', LABEL_SRC), ('label_trg', LABEL_TRG)])
"""

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
print(batch.label_src.shape)
print(batch.trg[0][:, 1:])
print(batch.trg[0])

class EncoderRNN(nn.Module):
  def __init__(self, emb_size, hidden_size, vocab_size, text_embedding_vectors, emotion_size, dropout=0):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    if text_embedding_vectors == None:
      self.embedding = nn.Embedding(vocab_size, emb_size)
    else:
      self.embedding = nn.Embedding.from_pretrained(
          embeddings=text_embedding_vectors, freeze=True)
    self.embedding_dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
    self.thought = nn.Linear(hidden_size*2+emotion_size, hidden_size)


  def forward(self, input_seq, emotion, hidden=None):
    embedded = self.embedding(input_seq) #[64, 30, 600]
    #embedded = self.embedding_dropout(embedded)
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
    self.concat_dropout = nn.Dropout(dropout)
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

    context = attn_weights.bmm(encoder_outputs) #[64, 1, 600]
    rnn_output = rnn_output.squeeze(1) #[64, 600]
    context = context.squeeze(1) #[64, 600]
    concat_input = torch.cat((rnn_output, context), 1) #[64, 1200]
    concat_input = self.concat_dropout(concat_input)
    concat_output = torch.tanh(self.concat(concat_input))
    output = self.out(concat_output)
    #output = F.softmax(output, dim=1)

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

def a_batch_loss(input_variable, target_variable, emotion, max_target_len, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, phase):
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
    for t in range(max_target_len-1):
      decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      decoder_input = target_variable[:, t] #[64], teaching_forceの場合、正解データを次に入力する
      decoder_output = F.softmax(decoder_output, dim=1)
      #loss += criterion(decoder_output, target_variable[:, t])
      #各バッチのtのlossをだす。mask_lossはnTotalで割った平均、nTotalはバッチ数からmask(<pad>)の数を引いたもの
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
    #total_loss += loss / max_target_len #1バッチ分のloss

  else:
    for t in range(max_target_len-1):
      decoder_output, decoder_hidden = decoder(
          decoder_input, decoder_hidden, encoder_outputs
      ) #[64, 単語種類数], [2, 64, 500]
      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([topi[i] for i in range(batch_size)])
      decoder_input = decoder_input.to(device)
      decoder_output = F.softmax(decoder_output, dim=1)
      #loss += criterion(decoder_output, target_variable[:, t])
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[:, t])
      loss += mask_loss
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
        #target_variable = batch.trg[0].to(device) #(64, 30)
        target_variable = batch.trg[0][:, 1:].to(device)
        emotion = batch.label_src
        emotion = label2one_hot(emotion)
        max_target_len = max(batch.trg[1])
        if target_variable.shape[0] == batch_size:
          total_loss = a_batch_loss(input_variable, target_variable, emotion, max_target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, phase) #1バッチ分のloss
          print_loss += total_loss #1epochのlossをprint_lossに加えていく

      #損失をだす
      print("epoch: {}; phase: {}; Average loss: {:.4f}; PPL: {:.4f}".format(epoch+1, phase, print_loss/i, math.exp(print_loss/i) ))

emb_size = 300
hidden_size = 600
dropout = 0.2
emotion_size = 7

clip = 1.0
teacher_forcing_ratio = 1.0
learning_rate = 0.002
decoder_learning_rate = 1.0
num_epochs = 10

#encoder = EncoderRNN(emb_size, hidden_size, len(SRC.vocab.stoi), None, emotion_size, dropout)
#decoder = LuongAttnDecoderRNN(emb_size, hidden_size, None, len(TRG.vocab.stoi),  dropout)

encoder = EncoderRNN(emb_size, hidden_size, len(SRC.vocab.stoi), SRC.vocab.vectors, emotion_size, dropout)
decoder = LuongAttnDecoderRNN(emb_size, hidden_size, TRG.vocab.vectors, len(TRG.vocab.stoi),  dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)

from torch import optim

dataloaders_dict = {"train": train_dl, "val": val_dl}

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate*decoder_learning_rate )

encoder.train()
decoder.train()

train_model(dataloaders_dict, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer)

"""
import pickle

model_path = '/content/drive/My Drive/dataset/DailyDialog/chatbot_encoder0809.pth'
torch.save(encoder.to('cpu').state_dict(), model_path)
model_path = '/content/drive/My Drive/dataset/DailyDialog/chatbot_decoder0809.pth'
torch.save(decoder.to('cpu').state_dict(), model_path)

model_path = '/content/drive/My Drive/dataset/DailyDialog/cuda_chatbot_encoder0809.pth'
torch.save(encoder.to('cuda').state_dict(), model_path)
model_path = '/content/drive/My Drive/dataset/DailyDialog/cuda_chatbot_decoder0809.pth'
torch.save(decoder.to('cuda').state_dict(), model_path)

with open('/content/drive/My Drive/dataset/DailyDialog/voc_word2index0809.pkl', 'wb') as f:
    pickle.dump(SRC.vocab.stoi, f)
with open('/content/drive/My Drive/dataset/DailyDialog/voc_index2word0809.pkl', 'wb') as f:
    pickle.dump(TRG.vocab.itos, f)
"""

encoder = EncoderRNN(emb_size, hidden_size, len(SRC.vocab.stoi), emotion_size, dropout)
decoder = LuongAttnDecoderRNN(emb_size, hidden_size, len(TRG.vocab.stoi),  dropout)
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

  def forward(self, input_seq, emotion, input_length, max_length):
    encoder_outputs, thought_vector = self.encoder(input_seq, emotion)
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

tempreture = 0.3

class GreedyTempreture(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedyTempreture, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, emotion, input_length, max_length):
    encoder_outputs, thought_vector = self.encoder(input_seq, emotion)
    cn = torch.zeros(1, 1, hidden_size).to(device)
    decoder_hidden = (thought_vector, cn)
    decoder_input = torch.LongTensor([TRG.vocab.stoi['<cls>']]).to(device)
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    for _ in range(max_length):
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)#[batch, vocab]
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

  def forward(self, input_seq, emotion, input_length, max_length):
    #seq2seq用
    encoder_outputs, thought_vector = self.encoder(input_seq, emotion)
    cn = torch.zeros(1, 1, hidden_size).to(device)
    hidden_list = (thought_vector, cn)
    #言語モデル用
    lm_hidden_list = (torch.zeros(1, 1, hidden_size).to(device), cn)
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

def evaluate(encoder, decoder, emotion, searcher, sentence, max_length=MAX_LENGTH):
    indexes_batch = indexesFromSentence(sentence)
    lengths = len(indexes_batch)
    input_batch = torch.LongTensor(indexes_batch).view(1, -1).to(device)
    tokens, scores = searcher(input_batch, emotion, lengths, max_length)
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
            inp = input('> ').lower()
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = inp.split('<emo>')[0]
            emotion = label2one_hot([int(inp.split('<emo>')[1])])
            #ノイズ処理
            input_sentence = normalizeString(input_sentence)
            output_words = evaluate(encoder, decoder, emotion, searcher, input_sentence)
            output_words[:] = [x for x in output_words if not (x == '<eos>' or x == '<pad>')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

encoder.eval()
decoder.eval()

batch_size = 1
#{ 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
#Oh, I’m sorry I bothered you. I’m really sorry.<emo>5


#searcher = GreedySearchDecoder(encoder, decoder)
#searcher = GreedyTempreture(encoder, decoder)
searcher = MMI(encoder, decoder)

evaluateInput(encoder, decoder, searcher)

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
    t = 0

    # Start beam search
    while True:
        #最大単語数を越したら
        if n_dec_steps > max_dec_steps:
            break

        # Fetch the best node
        score, _, n = heappop(nodes)
        decoder_input = n.wid #(1)
        print(t, TRG.vocab.itos[decoder_input])
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
            print(t, logp ,TRG.vocab.itos[decoded_t])
            node = BeamSearchNode(h=decoder_hidden,
                                  prev_node=n,
                                  wid=decoded_t,
                                  logp=n.logp+logp,
                                  length=n.length+1)
            heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps += beam_width
        t = t+1

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

def my_beam_search(decoder, encoder_output, thought_vector, beam_width, sos_token, eos_token, max_dec_steps, device):
    #assert beam_width >= n_best

    n_best_list = []

    cn = torch.zeros(1, 1, 600).to(device)
    decoder_hidden = (thought_vector, cn) #((1,1,600), (1,1,600))
    #<cls>を作成
    decoder_input = torch.LongTensor([sos_token]).to(device) #[1]
    end_nodes = []
    node = BeamSearchNode(h=decoder_hidden, prev_node=None, wid=decoder_input, logp=0, length=1)
    nodes = []
    #初期状態を追加
    heappush(nodes, (-node.eval(), id(node), node))
    n_dec_steps = 0
    #eosに到達した文を数える
    eos_count = 0

    #Repeat
    for t in range(max_dec_steps):
        #全ての候補文がeosなら終わり
        if len(end_nodes) >= beam_width:
            break
        t_list = []
        #beam幅分の候補で考える
        for _ in range(len(nodes)):
            #一番スコアの高い文章をだす
            score, _, n = heappop(nodes)
            decoder_input = n.wid #単語ID
            decoder_hidden = n.h  #LSTMに入れる隠れ状態と記憶せる

            #<eos>出力されたらこれ以上は出力しない
            if n.wid.item() == eos_token and n.prev_node is not None:
                if (score, id(n), n) not in end_nodes:
                      end_nodes.append((score, id(n), n))
                continue
            #候補文をデコーダに
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            topk_log_prob, topk_indexes = torch.topk(decoder_output, beam_width)
            #t時刻目のbeam幅分の候補単語とindexをまとめていく
            t_list.append((topk_log_prob, topk_indexes, n))

        #t時刻目の候補単語beam_width個を使ってそれぞれ出力候補を確認
        for topk_log_prob, topk_indexes, n in t_list:
            for new_k in range(beam_width):
                decoded_t = topk_indexes[0][new_k].view(1) # (1) new_k番目のindexを取り出す
                logp = topk_log_prob[0][new_k].item() # new_k番目の単語の生成確率を取り出す
                node = BeamSearchNode(h=decoder_hidden,
                                        prev_node=n,
                                        wid=decoded_t,
                                        logp=n.logp+logp,
                                        length=n.length+1)
                heappush(nodes, (-node.eval(), id(node), node))
        #nodesの中からbeam_width個を選択して次の時刻へ
        nodes = nodes[:beam_width]

    # if there are no end_nodes, retrieve best nodes (they are probably truncated)
    #if len(end_nodes) == 0:
    #   end_nodes = [heappop(nodes) for _ in range(beam_width)]
    end_nodes = nodes

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

beam_width = 5
sos_token = TRG.vocab.stoi['<cls>']
eos_token = TRG.vocab.stoi['<eos>']
max_dec_steps = 30

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

n_best_list = my_beam_search(decoder, encoder_outputs, thought_vector, beam_width, sos_token, eos_token, max_dec_steps, device)
print_n_best(n_best_list[0], TRG.vocab.itos)
