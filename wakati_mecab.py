from wakati import Wakati
import MeCab
import tqdm


def wakati_text(text):
    wakati = []
    mecab = MeCab.Tagger("-Ochasen")
    node = mecab.parseToNode(text)
    while node:
        hinshi = node.feature.split(",")[0]
        wakati.append(node.surface)
        node = node.next
    l = len(wakati)
    wakati = wakati[1:l-1]
    return ' '.join(wakati)


def add_wakati(wakati, path, texts):
    with open(path, 'a', encoding='utf-8') as f:
        if not wakati in texts:
            print(wakati, file=f)


def fix():
    pass


if __name__ == "__main__":
    w = Wakati()
    path = input('path> ')
    files = w.search_file(path)
    for f in tqdm.tqdm(files):
        lines = w.load_file(f)
        for line in lines:
            wakati = wakati_text(line)
            add_wakati(wakati, w.WAKATI, w.logs)
