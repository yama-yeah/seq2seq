from collections import defaultdict
path = 'wakati/wakati.txt'
with open(path, 'r', encoding='utf-8') as f:
    texts = [x.strip('\n') for x in f.readlines() if x]
word_index = defaultdict(lambda: 0)
for text in texts:
    words = text.split(' ')
    for word in words:
        word_index[word] += 1
print(len(word_index.keys()))
for text in texts:
    words = text.split(' ')
    l = len(words)
    for i in range(l):
        if word_index[words[i]] < 3:
            words[i] = 'UNK'
    with open('wakati/wakati_unk.txt', 'a', encoding='utf-8') as f:
        f.write(' '.join(words)+'\n')
