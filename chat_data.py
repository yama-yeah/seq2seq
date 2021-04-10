import numpy as np
import pickle
import copy
import morph
from w2v import W2v


class Chat_data:
    BATCH_SIZE = 1024
    ENDMARK = 'EOS'

    def __init__(self):
        self.w2v = W2v()

    def make_word_dictionary(self):
        # 文字辞書の作成
        self.index = self.w2v.get_index()
        # {文字char:要素数i}
        self.reverse_index = dict(
            (i, word) for word, i in self.index.items())
        self.matrix = self.w2v.get_embedding_matrix(self.index)
        self.num_tokens, self.w2v_size = self.matrix.shape

    def load_texts(self, time,  split_char):
        # コーパスをロード
        self.input_texts = []
        self.target_texts = []
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.max_seq_length = 0
        l = len(self.lines)
        for line in self.lines[self.BATCH_SIZE*(time-1): min(self.BATCH_SIZE*time, l)]:
            input_text, target_text = line.split(split_char)
            target_text = target_text + ' ' + self.ENDMARK
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            l = len(input_text.split(' '))
            if self.max_encoder_seq_length < l:
                self.max_encoder_seq_length = l
            l = len(target_text.split(' '))
            if self.max_decoder_seq_length < l:
                self.max_decoder_seq_length = l
        if self.max_decoder_seq_length < self.max_encoder_seq_length:
            self.max_seq_length = self.max_encoder_seq_length
        else:
            self.max_seq_length = self.max_decoder_seq_length

    def list2numpy(self, target_list):
        # メモリを節約するため
        return np.array(target_list)

    def text2onehot(self, text):
        input_data = copy.deepcopy(self.encoder_input_data)
        text = text.strip()
        i_want_sleep = morph.analyze(text, 1)
        for t, w in enumerate(i_want_sleep):
            print(w)
            # t=time(文字の出た時の要素数)
            # one-hot表現
            input_data[0, t, self.index[w[0]]] = 1.
        return input_data

    def create_input(self):
        self.encoder_input_data = np.zeros(
            (self.BATCH_SIZE, self.max_seq_length),
            dtype='float32')
        #[[[0]*num_encoder_tokens]* max_encoder_seq_length]*len(input_texts)
        self.decoder_input_data = np.zeros(
            (self.BATCH_SIZE, self.max_seq_length),
            dtype='float32')
        self.decoder_target_data = np.zeros(
            (self.BATCH_SIZE, self.max_seq_length, self.num_tokens),
            dtype='float32')
        return self.encoder_input_data, self.decoder_input_data, self.decoder_target_data

    def create_encoder_input(self, text):
        tokens = morph.analyze(text, 0)
        words = []
        for word, part in tokens:
            words.append(word)
        input_data = np.zeros(
            (1, len(words)),
            dtype='float32')
        for t, word in enumerate(words):
            input_data[0][t] = self.index[word]
        return input_data

    def make_train_data(self, time, split_char):
        self.load_texts(time, split_char)
        self.create_input()
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            input_text = input_text.split(' ')
            for t, word in enumerate(input_text):
                self.encoder_input_data[i][t] = self.index[word]
            target_text=target_text.split(' ')
            for t, word in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                if word == self.ENDMARK:
                    pass
                elif t > 0:
                    self.decoder_input_data[i][t +
                                               1] = self.index[word]
                else:
                    self.decoder_input_data[i][t] = self.index[self.ENDMARK]
                    self.decoder_input_data[i][t +
                                               1] = self.index[word]

                    #
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                self.decoder_target_data[i, t,
                                         self.index[word]] = 1.
        return [self.encoder_input_data, self.decoder_input_data, self.decoder_target_data]

    def test_data(self, time, split_char):
        self.load_texts(time, split_char)
        return self.input_texts

    def save(self):
        datas = {}
        datas['tokens'] = self.num_tokens
        datas['dics'] = [self.index, self.reverse_index]
        datas['matrix'] = self.matrix
        datas['size'] = self.w2v_size
        # datas['len']=[self.num_encoder_tokens,self.num_tokens]
        with open('pickle/chat.pickle', mode='wb') as f:
            pickle.dump(datas, f, protocol=2)

    def load(self):
        with open('pickle/chat.pickle', mode='rb') as f:
            datas = pickle.load(f)
        self.num_tokens = datas['tokens']
        self.index, self.reverse_index = datas['dics']
        return datas
        
    def load_file(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = f.read().splitlines()


if __name__ == "__main__":
    d = Chat_data()
    d.load_file('dataset/chat.txt')
    d.make_word_dictionary()
    d.save()
