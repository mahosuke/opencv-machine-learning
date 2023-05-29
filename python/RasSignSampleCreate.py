# -*- coding: utf-8 -*-
# 基本図形の交通標識: ./pictureSTD/*.png   11枚(50 x 50 pix)
# 基本図形の道路標識を、横方向に圧縮、縦方向に圧縮した変形図形を作成
# 画像サンプルを多くするため、回転、拡大で水増しする
# サンプル画像を出力するフォルダは、"./image/sign"（ランダムに抜き取り）
# データは、"./image/RasSignImg.npz"に保存
import os, glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2, random
import time

def transDX(signList):
    # 変形した標準画像を出力するフォルダ
    if not os.path.exists("./pictureDX"):
        os.makedirs("./pictureDX")
    for no, fname in enumerate(signList):
        signName = os.path.basename(fname)        # フォルダー名無しにする
        signName = signName[:-4]    # 拡張子除き
        classNo = int(signName[:2])    # 先頭2文字がクラス番号
        signImg = cv2.imread(fname)        #BGR/ndarray        ファイル名はフルネームで指定
        baseH, baseW = signImg.shape[:2]        #入力画像のサイズ
        fname = "./pictureDX/{}-ORG.png".format(signName)    # 標準図形も./pictureDX/に保存
        cv2.imwrite(fname, signImg)    # BGR(ndarray)
        # 標準図形の標識の変形（横、縦を圧縮）
        for sc in range(5, 55, 5):        #0% ～ 50%圧縮
            signImgDX = cv2.resize(signImg, (int(baseW * (100 - sc) / 100), baseH))    # ndarray
            signH, signW = signImgDX.shape[:2]
            wx = (baseW - signW) // 2
            signImgX = np.full((baseH, baseW, 3), 255, dtype=np.uint8)        # 白の正方形(H=ww,W=ww, color=3)画像をつくる。
            signImgX[:, wx:wx+signW] = signImgDX # 中央に標識画像をコピー
            fname = "./pictureDX/{}-X{}.png".format(signName, sc)
            cv2.imwrite(fname, signImgX)        # BGR(ndarray)
            signImgDY = cv2.resize(signImg, (baseW, int(baseH * (100 - sc) / 100)))
            signH, sjgnw = signImgDY.shape[:2]
            wy = (baseH - signH) // 2
            signImgY = np.full((baseH, baseW, 3), 255, dtype=np.uint8)        # 白の正方形(H=ww,W=ww, color=3)画像をつくる。
            signImgY[wy:wy+signH, :] = signImgDY # 中央に標識画像をコピー
            fname = "./pictureDX/{}-Y{}.png".format(signName, sc)
            cv2.imwrite(fname, signImgY)    # BGR(ndarray)

def sampleCreate(signDxList, imageSize):
    # 水増し変形標識画像の作成
    X, Y = [], []
    if not os.path.exists("./image/sign"):    # ランダムに選んだサンプル画像を保存
        os.makedirs("./image/sign")
    seqNo = list(range(len(signDxList)))
    random.shuffle(seqNo)        # 変形した図形の並びをシャッフル
    for no, picNo in enumerate(seqNo):
        fname = signDxList[picNo]
        signName = os.path.basename(fname)
        classNo = int(signName[:2]) 
        signImg = cv2.imread(fname)        # ndarray(GBR)
        # 回転、縮小/拡大する：PIL画像で行うため、ndarray->PILに変換
        baseImg = Image.fromarray(np.uint8(signImg))    # numpy 配列画像を、PIL画像に変換
        for ang in range(-20, 22, 2):    # 回転-20, -18,....0, ...18, 20deg
            subImg = baseImg.rotate(ang, fillcolor=(255, 255, 255))    #回転後の隙間を白にする
            data = np.asarray(subImg)    # asarrayなので、data = subImgとなる
            X.append(data)
            Y.append(classNo)
            w = imageSize
            for ratio in range(8, 15, 3):    # 縮小・拡大する(70%, 100%, 130% , 160%)
                size = round((ratio/10) * imageSize)
                img2 = cv2.resize(data, (size, size), cv2.INTER_AREA)
                data2 = np.asarray(img2)
                if imageSize > size:    # 変形画像が小さい時は空白画像の中心にコピー
                    x = (imageSize - size) // 2
                    data = np.full((imageSize, imageSize, 3), 255, dtype=np.uint8)        # 白の正方形(H=ww,W=ww, color=3)画像をつくる。
                    data[x:x+size, x:x+size] = data2
                else:                            # 変形画像が大きい時は、変形画像から定型サイズを切り抜く
                    x = (size - imageSize) // 2
                    data = data2[x:x+w, x:x+w]
                X.append(data)
                Y.append(classNo)
                # 参考にサンプリングで画像データを保存(400回に１回）
                if random.randint(0, 400) == 0:
                    fname = "image/sign/{0}({1})({2})({3}).png".format(signName[:-4], classNo, ang, ratio)
                    cv2.imwrite(fname, data)
    return X, Y

"""メイン関数"""
if __name__ == '__main__':
    # サイズの指定
    imageSize = 50 # 50x50
    # 標準標識の画像(50x50pixel)ファイルリスト。ファイル名={クラス番号}{標識名}-STD.png
    signList = glob.glob("./pictureSTD/*.png") 
    transDX(signList)    # 縦・横を#0% ～ 50%圧縮
    # 変形を含めた図形を、回転、縮小・拡大で水増しして、Ｘ、Ｙに加える
    signDxList = glob.glob("./pictureDX/*.png")        # 変形した図形 リスト
    X, Y = sampleCreate(signDxList, imageSize)
    X = np.array(X)
    Y = np.array(Y)
    np.savez("./image/RasSignImg.npz", x=X, y=Y)
    print("ok,", len(Y))