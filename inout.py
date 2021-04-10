path = 'wakati/wakati_unk.txt'
with open(path, 'r', encoding='utf-8') as f:
    texts = [x.strip('\n') for x in f.readlines() if x]
l = len(texts)
inp = []
out = []
l = l-1
for i in range(l):
    inp.append(texts[i])
    out.append(texts[i+1])
with open('dataset/chat.txt', 'w', encoding='utf-8') as f:
    for i in range(l):
        line = inp[i]+'\t'+out[i]+'\n'
        f.write(line)
