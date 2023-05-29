# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn import model_selection
import numpy as np
import time
import os
imgW, imgH = 50, 50
nClass = 11

def buildModelCNN():
    # CNNのモデルを構築
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(imgW, imgH, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)),
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nClass, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])
    return model

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    signFileName = "./image/RasSignImg.npz"
    if not os.path.exists(signFileName):
        print("Sign File does not exist")
        exit()
    saveModelFile = "./RasSignModelCNN_1.h5"
    sTime = time.time()
    # フォント画像のデータを読む
    xy = np.load(signFileName)
    X = xy["x"]
    Y = xy["y"]
    # データを正規化
    X = X.reshape(X.shape[0], imgW, imgH, 3).astype('float32')
    X /= 255
    Y = np_utils.to_categorical(Y, nClass)
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
    # モデルを構築
    model = buildModelCNN()
    history = model.fit(X_train, y_train,
                                    batch_size=128, epochs=5, verbose=1,
                                    validation_data=(X_test, y_test))
    # モデルを保存
    model.save(saveModelFile)
    model.summary()
    # モデルを評価
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss = {},  Test Accuracy = {}".format(score[0], score[1]))
    print('Conputation Time = ', time.time() - sTime)
    # 学習過程をグラフ化
    plt.figure(1, figsize = (12, 3))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'black', label='training')
    plt.plot(history.history['val_loss'], 'red', label='test')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], 'black', label='training')
    plt.plot(history.history['val_accuracy'], 'red', label='test')
    plt.legend()
    plt.show()

