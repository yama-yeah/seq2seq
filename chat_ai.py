
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.utils import plot_model  # モデル保存ライブラリーのインポート 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding,  Attention, Concatenate, Bidirectional,Dropout,BatchNormalization
import numpy as np

from chat_data import Chat_data

from w2v import W2v


from tensorflow.keras.models import load_model

from janome import tokenizer

#from attention import SimpleAttention as sa
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow.keras.backend as K


import pathlib

current_dir = str(pathlib.Path(__file__).resolve().parent)

class Network:
    batch_size = 8  # Batch size for training.
    epochs = 64  # Number of epochs to train for.
    latent_dim = 512  # Latent dimensionality of the encoding space.
    SAVE_PATH = current_dir+'/weight/'
    MAX = 256
    # Path to the data txt file on disk.

    def __init__(self):
        self.dataset = Chat_data()
        self.w2v = W2v()

    def load_datas(self):
        datas = self.dataset.load()

        self.num_tokens = datas['tokens']
        self.index, self.reverse_index = datas['dics']
        self.w2v_size = datas['size']
        self.matrix = datas['matrix']
    '''
    def load_input(self):
        self.encoder_input_data,self.decoder_input_data,self.decoder_target_data=self.dataset.create_input()
    '''

    def load_w(self, model):
        model.load_weights(self.SAVE_PATH+'cw.h5')

    def save_w(self, model):
        model.save_weights(self.SAVE_PATH+'cw.h5')

    def study_model(self):
        self.encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        # 入力層
        self.embedding = Embedding(input_dim=(
            self.num_tokens), output_dim=self.w2v_size, weights=[self.matrix], mask_zero=True,trainable=False)
        # 埋め込み、単語を128次元ベクトル化 ０をパンディング
        e_inputs = self.embedding(self.encoder_inputs)
        e_inputs = Dropout(rate=0.2)(e_inputs)
        encoder = Bidirectional(LSTM(int(self.latent_dim/2), return_state=True,return_sequences=True,dropout=0.5,recurrent_dropout=0.5),name="Bidirectional_encoder_LSTM")
        # latent_dim 出力の次元数．
        # 真理値．出力とともに，最後の状態を返すかどうか．
        encoder_outputs, f_h, f_c,b_h,b_c = encoder(e_inputs)
        encoder_outputs=BatchNormalization()(encoder_outputs)
        f_h=BatchNormalization()(f_h)
        f_c=BatchNormalization()(f_c)
        b_h=BatchNormalization()(b_h)
        b_c=BatchNormalization()(b_c)
        state_h=Concatenate(name="state_h_Concat")([f_h,b_h])
        state_c=Concatenate(name='state_c_Concat')([f_c,b_c])
        # encoder_outputsは使わない
        self.encoder_states = [state_h, state_c]
        self.normal_out_encoder = [encoder_outputs, state_h, state_c]
        self.decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        d_inputs = self.embedding(self.decoder_inputs)
        d_inputs = Dropout(rate=0.2)(d_inputs)
        self.decoder_lstm = LSTM(
            self.latent_dim, return_sequences=True, return_state=True,dropout=0.5,recurrent_dropout=0.5,name='decoder_LSTM')
        decoder_outputs, _, _ = self.decoder_lstm(d_inputs,
                                                  initial_state=self.encoder_states)
        decoder_outputs=BatchNormalization()(decoder_outputs)
        #self.attention_layer=sa(self.latent_dim)
        #decoder_outputs=self.attention_layer(decoder_outputs,encoder_outputs)
        self.attention_layer=Attention(self.latent_dim)
        a_w=self.attention_layer([decoder_outputs,encoder_outputs])
        # initial_state encoderの隠れ状態を引き渡す
        # 全結合ニューラルネットワークの定義(softmaxで単語の確率を出してる)
        # 活性化関数softmax
        # 出力空間の次元数 num_decoder_tokens(文字の種類の総数)
        self.concat=Concatenate()
        decoder_outputs=self.concat([decoder_outputs,a_w])

        self.decoder_dense = Dense(
            self.num_tokens, activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)
        model = Model(
            [self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        # モデルの入力、出力の形
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        # rmsprop 勾配 RMSprop
        # loss 	損失関数 クロスエントロピー
        return model

    def encoder_model(self):
        encoder = Model(self.encoder_inputs, self.normal_out_encoder)
        return encoder

    def decoder_model(self):
        decoder_state_input_h = Input(
            shape=(self.latent_dim,), name='decoder_state_h')
        decoder_state_input_c = Input(
            shape=(self.latent_dim,), name='decoder_state_c')
        encoder_outputs = Input((None,None), name='encoder_outputs')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        d_inputs = self.embedding(self.decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            d_inputs, initial_state=decoder_states_inputs)
        decoder_outputs=BatchNormalization()(decoder_outputs)
        decoder_states = [state_h, state_c]
        a_w = self.attention_layer([decoder_outputs,encoder_outputs])
        decoder_outputs = self.concat([decoder_outputs,a_w])
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder = Model(
            [self.decoder_inputs, decoder_state_input_h, decoder_state_input_c,encoder_outputs],
            [decoder_outputs, state_h, state_c])
        return decoder
    # decoder側のモデル

    def responder(self, input_seq):
        # Encode the input as state vectors.
        e_out, h, c = self.encoder.predict(input_seq)
        # 入力サンプルに対する予測値の出力を生成します．
        # input_seq  入力データ

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        sampled_char = 'EOS'
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict(
                [np.array([[self.index[sampled_char]]], 'float32'), h, c,e_out])
            # output_tokens=self.attention_layer.call([e_out,output_tokens])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_index[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == 'EOS' or
                    len(decoded_sentence) > self.MAX):
                break

            decoded_sentence += sampled_char
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def create_model(self):
        study = self.study_model()
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        return study

    def test(self):
        #self.encoder_input_data, _, _ = self.dataset.make_train_data(1, '\t')
        input_texts = self.dataset.input_texts
        for seq_index in range(self.dataset.BATCH_SIZE):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = self.encoder_input_data[seq_index:seq_index + 1]
            decoded_sentence = self.responder(input_seq)

            print('-')
            print('Input sentence:', input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)

    def load_train(self, time, split):
        self.encoder_input_data, self.decoder_input_data, self.decoder_target_data = self.dataset.make_train_data(
            time, split)

    def draw_model(self, model, name):
        model.summary()  # スクリーンにモデル表示
        plot_model(model, to_file=name+".png", show_shapes=True)  # モデル保存実行

    def train(self, model):
        '''
        print(np.shape(self.encoder_input_data))
        print(np.shape(self.decoder_target_data))
        print(np.shape(self.decoder_input_data))
        '''
        model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[EarlyStopping(patience=16)], 
                  validation_split=0.25)
        # 学習
        # [encoder_input_data, decoder_input_data] 訓練データ
        # decoder_target_data 教師データ
        # バッチサイズ
        # epochs 訓練データ配列の反復回数を示す整数
        # validation_split ニューラルネットワークのハイパーパラメータの良し悪しを確かめるための検証データ。学習は行わない。

    def ez_response(self, text):
        input_data = self.dataset.create_encoder_input(text)
        response = self.responder(input_data)
        return response


if __name__ == "__main__":
    
    net = Network()
    net.load_datas()
    net.dataset.load_file('dataset/chat.txt')
    model = net.create_model()
    
    net.draw_model(model,'study')
    net.draw_model(net.encoder,'encoder')
    net.draw_model(net.decoder,'decoder')
    
    #net.load_w(model)

    for i in range(int(82737*95/100/net.dataset.BATCH_SIZE)):
        try:
            net.load_train(1+i, '\t')
            # net.draw_model(model,'study')
            net.train(model)
            net.save_w(model)
        except:
            pass
            
    """
    for i in range(5):
        try:
            net.load_train(1+i, '\t')
            # net.draw_model(model,'study')
            net.train(model)
        except:
            break"""
    
    net.load_train(int(82737*95/100/net.dataset.BATCH_SIZE)+1, '\t')
    # net.draw_model(model,'study')
    #net.train(model)
    #net.load_train(i+2,'\t')
    net.test()
