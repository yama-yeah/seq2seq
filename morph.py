import re
from janome.tokenizer import Tokenizer
TOKENNIZER = Tokenizer()


def analyze(text, flag):
    # textを形態素解析し[(surface,parts)]の形にして返す リスト型
    # 表層系、品詞
    if flag == 0:
        return [(t.surface, t.part_of_speech) for t in TOKENNIZER.tokenize(text)]
    else:
        return [(t.surface, t.part_of_speech, t.reading) for t in TOKENNIZER.tokenize(text)]


def is_keyword(part):
    return bool(re.match(r'名詞,(一般|代名詞|固有名詞|サ変接続|形容動詞語幹)', part))


def fix(text):
    return ''.join([token.surface for token in TOKENNIZER.tokenize(
        text) if not token.part_of_speech.split(',')[0] == '記号'])


if __name__ == "__main__":
    text = input('>')
    print(analyze(text, 1))
