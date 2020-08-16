import pandas as pd
import MeCab
import unicodedata
import re

mecab = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

#mecab.parse(text)
def length_cut(input, output):
    return 5<=len(input)<=20 and 5<=len(output)<=20

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
    s = re.sub(r'[^a-zA-Zあ-ん一-龥0-9、。,.!? ]', '', s)
    return s

def make(path):
    with open(path + '.txt') as f:
        #txtデータの読み込み
        data = f.readlines()
        inp = []
        opt = []

        for i in range(0, len(data)-1, 2):
            inp_parse = mecab.parse(normalizeString(data[i])).split()
            opt_parse = mecab.parse(normalizeString(data[i+1])).split()
            if length_cut(inp_parse, opt_parse):
                inp.append(' '.join(inp_parse))
                opt.append(' '.join(opt_parse))
        df = pd.DataFrame({0:inp, 1:opt})
        df = df.drop_duplicates()
        df = df.sample(frac=1, random_state=13)
        print('len:', len(df))
        df[:-10000].to_csv('train_' + path + '.csv', header=None, index=None)
        df[-10000:].to_csv('val_' + path + '.csv', header=None, index=None)

paths = ['tiu_twitter']
for path in paths:
    make(path)
