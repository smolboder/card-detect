import cv2 as cv
from glob import glob
import os
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import model_from_json


def reorder(approx):
    """重新排列approx的資料順序 之後opencv畫四邊形會用到"""
    approx = np.resize(approx, (4, 2))
    new = np.zeros_like(approx)
    add = approx.sum(1)
    diff = np.diff(approx, axis=1)
    new[0] = approx[np.argmin(add)]
    new[3] = approx[np.argmax(add)]
    new[1] = approx[np.argmin(diff)]
    new[2] = approx[np.argmax(diff)]
    return new


def findCon(img):
    """用opencv 從攝影機畫面中擷取撲克牌的影像"""
    imgBlur = cv.GaussianBlur(img, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 150, 150)
    imgDilate = cv.dilate(imgCanny, (5, 5), 3)
    imgErod = cv.dilate(imgDilate, (5, 5), 1)
    contours, heirarchy = cv.findContours(imgErod, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    data = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        pere = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.01 * pere, True)
        approx = reorder(approx)
        data.append([area, cnt, approx])

    data = sorted(data, key=lambda x: x[0], reverse=True)
    maxdata = data[0]

    a, b, c = maxdata[2][0], maxdata[2][1], maxdata[2][2]
    w = int(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
    h = int(((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2) ** 0.5)

    pts1 = np.float32(maxdata[2])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv.warpPerspective(img, M, (w, h))
    return imgWarp, maxdata


if __name__=='__main__':

    img_file = []
    className = []
    dict = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}
    '''紅鬼 53  黑鬼 54  廠商資訊 55'''
    # 以55張撲克牌圖片檔案名稱 依序編號為0~54 並新增到到className串列 作為輸入CNN模型的資料分類
    for file in glob('./Card/*/*'):
        img_file.append(file)
        file_basename = os.path.basename(file)
        name = os.path.splitext(file_basename)[0]
        a = re.search('([A-Za-z]*)_(.*)_', name)
        if a.group(1) == 'Spade':
            txt = dict[a.group(2)]
        elif a.group(1) == 'Heart':
            txt = dict[a.group(2)] + 13 * 1
        elif a.group(1) == 'Club':
            txt = dict[a.group(2)] + 13 * 3
        elif a.group(1) == 'Diamond':
            txt = dict[a.group(2)] + 13 * 2
        elif a.group(1) == 'Ghost' and a.group(2) == 'red':
            txt = 53
        elif a.group(1) == 'Ghost' and a.group(2) == 'black':
            txt = 54
        else:
            txt = 55
        txt = txt-1
        className.append(txt)

    train_x = np.array([cv.imread(x) for x in img_file])
    train_y = np.array([x for x in className])

    print(train_x.shape)
    print(train_y.shape)
    print('資料讀取完成')

    category = 55
    train_x = np.array(list(map(lambda x: cv.resize(findCon(x), (125, 125)), train_x)))

    # 數據標準化
    train_x = train_x.astype('float32')
    train_x = train_x/255

    train_y = tf.keras.utils.to_categorical(train_y,num_classes=category)
    print(train_x.shape)
    print(train_y[0])
    # 模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=3,
                                     kernel_size=(3,3),
                                     activation='relu',
                                     padding='same',
                                     input_shape=(125,125,3)))
    model.add(tf.keras.layers.Conv2D(filters=9,
                                     kernel_size=(3,3),
                                     activation='relu',
                                     padding='same',
                                     input_shape=(125,125,3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128,
                                    activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(units=128,
                                    activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(units=category,
                                    activation=tf.nn.softmax))

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,
                                                      epsilon=1e-03,schedule_decay=0.005),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    # 訓練
    history = model.fit(train_x, train_y, batch_size=20, epochs=100, verbose=1)

    # 存檔 模型和訓練權重
    with open('card.json','w') as json_file:
        json_file.write(model.to_json())
    model.save_weights('card.h5')








