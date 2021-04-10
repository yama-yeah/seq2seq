from janome.tokenizer import Tokenizer
import os
import morph
import glob
import tqdm


# ' 'を使って分かつ

class Wakati():
    WAKATI = 'wakati/wakati.txt'

    def __init__(self):
        try:
            with open(self.WAKATI, 'r', encoding='utf-8') as f:
                self.texts = [x.strip() for x in f.readlines() if x]
        except:
            open(self.WAKATI, 'w').close
            self.texts = ''

    def wakati_text(self, text):
        t = Tokenizer()
        return t.tokenize(text, wakati=True)

    def add_wakati(self, wakati):
        wakati = ' '.join(wakati)
        with open(self.WAKATI, 'a', encoding='utf-8') as f:
            if not wakati in self.texts:
                print(wakati, file=f)

    def load_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
        return lines

    def search_file(self, path):
        files = glob.glob(path+'*.txt')
        return files

    @property
    def logs(self):
        return self.texts


if __name__ == '__main__':
    # 一気に分かちたいとき
    w = Wakati()
    path = input('path> ')
    files = w.search_file(path)
    for f in tqdm.tqdm(files):
        lines = w.load_file(f)
        for line in lines:
            line = morph.fix(line)
            wakati = w.wakati_text(line)
            w.add_wakati(wakati)
