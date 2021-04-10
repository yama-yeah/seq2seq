from gensim.models import word2vec
import random
import morph
import numpy as np
MODEL = 'weight/w2v_model.model'
WAKATI = 'wakati/wakati_unk.txt'


class W2v:
    def __init__(self):
        self.model = word2vec.Word2Vec.load(MODEL)

    def load_w2v(self, word):
        model = word2vec.Word2Vec.load('w2v.model')
        try:
            # print('w2v')
            similar_words = model.most_similar(positive=[word])
            return random.choice([w[0] for w in similar_words])
        except:
            return word

    def make_model(self):
        w2v_data = word2vec.LineSentence(WAKATI)
        model = word2vec.Word2Vec(
            w2v_data, size=256, window=3, hs=1, min_count=0, sg=1)
        model.save(MODEL)

    def word_calculator(self, text, parts):
        pass

    def get_embedding_matrix(self,word_index):
        """
        tf.keras.layers.Embeddingのweights引数で指定するための重み行列作成
        model: gensim model
        num_word: modelのvocabularyに登録されている単語数
        emb_dim: 分散表現の次元
        word_index: gensim modelのvocabularyに登録されている単語名をkeyとし、token idをvalueとする辞書 ex) {'word A in gensim model vocab': integer token id} 
        """
        # gensim modelの分散表現を格納するための変数を宣言
        embedding_matrix = np.zeros(
            (max(list(word_index.values()))+1, self.model.vector_size), dtype="float32")

        # 分散表現を順に行列に格納する
        for word, label in word_index.items():
            # labelは数字
            try:
                # gensimのvocabularyに登録している文字列をembedding layerに入力するone-hot vectorのインデックスに変換して、該当する重み行列の要素に分散表現を代入
                embedding_matrix[label] = self.model.wv[word]
            except KeyError:
                print('error:'+word)
        return embedding_matrix

    def get_index(self):
        word_list = self.model.wv.index2word
        word_index = {}
        for i, word in enumerate(word_list):
            word_index[word] = i+2
        word_index['EOS']=1
        return word_index


if __name__ == "__main__":
    w2v = W2v()
    w2v.make_model()
    print(w2v.get_index()['EOS'])
    #word_index = {key: int(key) for key in model.wv.vocab}
    #e = get_embedding_matrix(model, word_index)
    # print(e)
