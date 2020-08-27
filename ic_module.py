#! -*- coding: utf-8 -*-
import glob
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

FileNames = ["img1.npy", "img2.npy", "img3.npy"]
ClassNames = ["うさぎ", "いぬ", "ねこ"]
hw = {"height":32, "width":32}        # リストではなく辞書型 中かっこで囲む


################################
###### 画像データの前処理 ######
################################
def PreProcess(dirname, filename, var_amount=3):
    num = 0
    arrlist = []
    files = glob.glob(dirname + "/*.jpeg")

    for imgfile in files:
        img = load_img(imgfile, target_size=(hw["height"], hw["width"]))    # 画像ファイルの読み込み
        array = img_to_array(img) / 255                                     # 画像ファイルのnumpy化
        arrlist.append(array)                 # numpy型データをリストに追加
        for i in range(var_amount-1):
            arr2 = array
            arr2 = random_rotation(arr2, rg=360)
            arrlist.append(arr2)              # numpy型データをリストに追加
        num += 1

    nplist = np.array(arrlist)
    np.save(filename, nplist)
    print(">> " + dirname + "から" + str(num) + "個のファイル読み込み成功")


################################
######### モデルの構築 #########
################################
def BuildCNN(ipshape=(32, 32, 3), num_classes=3):
    model = Sequential()

    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape))
    model.add(Activation('relu'))

    model.add(Conv2D(48, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


################################
############# 学習 #############
################################
def Learning(tsnum=30, nb_epoch=50, batch_size=8, learn_schedule=0.9):
    X_TRAIN_list = []; Y_TRAIN_list = []; X_TEST_list = []; Y_TEST_list = [];
    target = 0
    for filename in FileNames :
        data = np.load(filename)          # 画像のnumpyデータを読み込み
        trnum = data.shape[0] - tsnum
        X_TRAIN_list += [data[i] for i in range(trnum)]          # 画像データ
        Y_TRAIN_list += [target] * trnum                         # 分類番号
        X_TEST_list  += [data[i] for i in range(trnum, trnum+tsnum)]          # 学習しない画像データ
        Y_TEST_list  += [target] * tsnum;                                     # 学習しない分類番号
        target += 1

    X_TRAIN = np.array(X_TRAIN_list + X_TEST_list)    # 連結
    Y_TRAIN = np.array(Y_TRAIN_list + Y_TEST_list)    # 連結
    print(">> 学習サンプル数 : ", X_TRAIN.shape)
    y_train = np_utils.to_categorical(Y_TRAIN, target)    # 自然数をベクトルに変換
    valrate = tsnum * target * 1.0 / X_TRAIN.shape[0]

    # 学習率の変更
    class Schedule(object):
        def __init__(self, init=0.001):      # 初期値定義
            self.init = init
        def __call__(self, epoch):           # 現在値計算
            lr = self.init
            for i in range(1, epoch+1):
                lr *= learn_schedule
            return lr

    def get_schedule_func(init):
        return Schedule(init)

    lrs = LearningRateScheduler(get_schedule_func(0.001))
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model = BuildCNN(ipshape=(X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]), num_classes=target)

    print(">> 学習開始")
    hist = model.fit(X_TRAIN, y_train,
                     batch_size=batch_size,
                     verbose=1,
                     epochs=nb_epoch,
                     validation_split=valrate,
                     callbacks=[lrs, mcp])

    json_string = model.to_json()
    json_string += '##########' + str(ClassNames)
    open('model.json', 'w').write(json_string)
    model.save_weights('last.hdf5')


################################
########## 試行・実験 ##########
################################
def TestProcess(imgname):
    modelname_text = open("model.json").read()
    json_strings = modelname_text.split('##########')
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    model = model_from_json(json_strings[0])
    model.load_weights("last.hdf5")  # best.hdf5 で損失最小のパラメータを使用
    img = load_img(imgname, target_size=(hw["height"], hw["width"]))    
    TEST = img_to_array(img) / 255

    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)
    print(">> 計算結果↓\n" + str(pred))
    print(">> この画像は「" + textlist[np.argmax(pred)].replace(",", "") + "」です。")
