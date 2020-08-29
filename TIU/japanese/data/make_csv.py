import pandas as pd
import MeCab
import unicodedata
import re
import numpy as np
import random

mecab = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

#mecab.parse(text)
def length_cut(input, output):
    return 3<=len(input)<=20 and 3<=len(output)<=20

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
    )

def normalizeString(text):
    if '【' in text:
        return ''
    s = unicodeToAscii(text.lower().strip())
    s = re.sub(r"([.!?])", r"\1", s)
    s = re.sub(r"\s+", r" ", s).strip()
    s = re.sub(r'[\r]', '', s)
    s = re.sub(r'　', ' ', s)                    #全角空白の除去
    s = re.sub(r'req:', '', s)
    s = re.sub(r'res:', '', s)
    s = re.sub(r'[^a-zA-Zぁ-んァ-ン一-龥0-9、。,.!?ー ]', '', s)
    return s

def make(path):
    ratio = 0.1
    with open(path + '.txt') as f:
        #txtデータの読み込み
        data = f.readlines()
        inp = []
        opt = []
        gomi = 0
        iiyo = 0

        for i in range(0, len(data)-1, 2):
            inp_parse = mecab.parse(normalizeString(data[i])).split()
            opt_parse = mecab.parse(normalizeString(data[i+1])).split()
            if length_cut(inp_parse, opt_parse):
                if opt_parse[0] == '私':
                    pass
                    
                else:
                    inp.append(' '.join(inp_parse))
                    opt.append(' '.join(opt_parse))

        df = pd.DataFrame({0:inp, 1:opt})
        print('len:', len(df))
        df = df.drop_duplicates()
        df = df.sample(frac=1, random_state=13)
        print('len:', len(df))
        df = df.dropna()
        print('len:', len(df))
        print('iiyo', iiyo)
        print('gomi', gomi)
        df[:-1000].to_csv('train_' + path + '.csv', header=None, index=None)
        df[-1000:].to_csv('val_' + path + '.csv', header=None, index=None)

paths = ['desumasu']
for path in paths:
    make(path)
