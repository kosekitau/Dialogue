import pandas as pd
import MeCab
import unicodedata
import re

#mecab.parse(text)
mecab = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

def length_cut(input, output):
    return 5<=len(mecab.parse(input).split())<=20 and 5<=len(mecab.parse(output).split())<=20

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
    )

def normalizeString(text):
    s = unicodeToAscii(text.lower().strip())
    s = re.sub(r"([.!?])", r"\1", s)
    s = re.sub(r"\s+", r" ", s).strip()
    s = re.sub(r'[【】]', '', s)                  # 【】の除去
    s = re.sub(r'[（）()]', '', s)                # （）の除去
    s = re.sub(r'[［］\[\]]', '', s)              # ［］の除去
    s = re.sub(r'[\r]', '', s)
    s = re.sub(r'　', ' ', s)                    #全角空白の除去
    s = re.sub(r'req:', '', s)
    s = re.sub(r'res:', '', s)
    s = re.sub(r'[^a-zA-Zぁ-んァ-ン一-龥0-9、。,.!?ー ]', '', s)
    return s

def make(path):
    with open(path + '.txt') as f:
        #txtデータの読み込み
        data = f.readlines()
        inp = []
        opt = []

        for i in range(0, len(data)-1, 2):
            inp_w = normalizeString(data[i])
            opt_w = normalizeString(data[i+1])
            if length_cut(inp_w, opt_w):
                inp.append(inp_w)
                opt.append(opt_w)
        df = pd.DataFrame({0:inp, 1:opt})
        df = df.drop_duplicates()
        df = df.sample(frac=1, random_state=13)
        print('len:', len(df))
        df[:-1000].to_csv('sp_train_' + path + '.csv', header=None, index=None)
        df[-1000:].to_csv('sp_val_' + path + '.csv', header=None, index=None)

paths = ['pre']
for path in paths:
    make(path)
