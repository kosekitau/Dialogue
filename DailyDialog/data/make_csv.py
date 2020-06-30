import pandas as pd

def length_cut(input, output):
    return 4 < len(input.split()) < 29 and 4 < len(output.split()) < 29

def make(path):
    with open(path + '/dialogues_' + path + '.txt') as f:
        data = f.readlines()
        inp = []
        opt = []
        for i in range(len(data)):
            d = data[i].replace('\n',' ').split(' __eou__ ')[:-1]
            for j in range(0, len(d)-1):
                if length_cut(d[j], d[j+1]):
                    inp.append(d[j])
                    opt.append(d[j+1])
        df = pd.DataFrame({0:inp, 1:opt})
        print('len:', path, len(df))
        df.to_csv(path + '.csv', header=None, index=None)

paths = ['train', 'validation', 'test']
for path in paths:
    make(path)
