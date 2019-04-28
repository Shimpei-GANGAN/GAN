#----------------------- -------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#	Updata History
#	January  18  06:00, 2019 (Fri) by S.Iwamaru
#------------------------------------------------------------
#
#	GANクラス。
#
#------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model


class GAN():
    def __init__(self):
        #  MNISTデータ用
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        #  潜在変数の次元数
        self.z_dim = 100
        
        optimizer = Adam(0.0002, 0.5)
        
        #  Discriminatorモデル
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=optimizer,
                                   metrics=["accuracy"])
        
        #  Generatorモデル
        self.generator = self.build_generator()
        #  generatorは学習しないからコメントアウト
        #self.generator.compile(loss="binary_crossentropy", optimizer=optimizer)
        
        #  モデルを接続
        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)
        
    def build_generator(self):
        noise_shape = (self.z_dim,)
        
        model = Sequential()
        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation="tanh"))
        model.add(Reshape(self.img_shape))
        
        model.summary()
        
        return model
        
    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()
        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()
        
        return model
        
    def build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model
        
    def build_combined2(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        
        model = Model(z, valid)
        model.summary()
        return model
    
    def train(self, epochs, batch_size=128, save_interval=50):
        #  mnistのデータを読み込み
        (X_train, _), (_, _) = mnist.load_data()
        
        #  -1 to 1に正規化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        half_batch = int(batch_size / 2)
        
        num_batches = int(X_train.shape[0] / half_batch)
        print("Number of batches:{}".format(num_batches))
        
        for epoch in range(epochs):
            for iteration in range(num_batches):
                # ---------------------
                #  Discriminatorの学習
                # ---------------------
                
                #  バッチサイズの半分をGeneratorから生成
                noise = np.random.normal( 0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)
    
                #  バッチサイズの半分を教師データから抽出
                idx = np.random.randint( 0, X_train.shape[0], half_batch)
                imgs = X_train[idx]
                
                #  本物と偽物を別々に学習
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                
                #  それぞれの損失関数を平均
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                
                # ---------------------
                #  Generatorの学習
                # ---------------------
                
                noise = np.random.normal( 0, 1, (batch_size, self.z_dim))
                
                #  生成データは正解ラベルを設定
                valid_y = np.array([1] * batch_size)
                
                #  Generatorの学習
                g_loss = self.combined.train_on_batch(noise, valid_y)
                
                #  進捗の表示
                print("epoch:{}, iter:{}, [D loss:{}, acc:{}][G loss:{}]".format( epoch, iteration, d_loss[0], 100 * d_loss[1], g_loss))
                
                #  一定間隔で生成画像を保存していく
                if epoch % save_interval == 0:
                    self.save_imgs(epoch)
                
    def save_imgs(self, epoch):
        r, c = 5, 5
        
        noise = np.random.normal( 0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        
        #  生成画像を再スケール
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i,j].axis("off")
                cnt += 1
        fig.savefig( "data/mnist_{}.png".format(epoch) )
        plt.close()
        
if __name__ == "__main__":
    gan = GAN()
    gan.train(epochs=100, batch_size=32, save_interval=1)
