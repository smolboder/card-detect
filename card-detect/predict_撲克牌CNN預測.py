import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2 as cv
import numpy as np
from 專案_物品辨識.撲克牌CNN import findCon
from PIL import Image, ImageDraw, ImageFont


def get_card_data(predict):
    """依照模型預測結果的分類(0~54)轉換為撲克牌資料"""
    color = predict // 13
    num = predict % 13
    if num == 0:
        color -= 1  # 假設predict是13 整除得到1 但必須是0才可以 因為1~13為一組
        num = 13
    if color == 0:
        txt = '黑桃'+str(num)
    elif color == 1:
        txt = '紅心'+str(num)
    elif color == 2:
        txt = '方塊'+str(num)
    elif color == 3:
        txt = '梅花'+str(num)
    else:
        if predict == 53:
            txt = '紅鬼'
        elif predict == 54:
            txt = '黑鬼'
        else:
            txt = '廠商資訊'
    return txt


# 讀取模型
json_file = open('card.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
json_file.close()
# 讀取權重
model.load_weights('card.h5')

# opencv 攝影機物件
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cv.namedWindow('imgWarp', 0)
cv.resizeWindow('imgWarp', 125, 125)
fontInfo = ImageFont.truetype('C:\Windows\Fonts\msjh.ttc', 36)  # 字形

'''openCV 建立儲存影片物件 沒有聲音'''
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv.VideoWriter('CardPredict.avi', fourcc, 20.0, (640,  480))

while 1:
    success, img = cap.read()
    if not success:
        print('讀取不到畫面')
        break

    try:
        imgWarp, maxdata = findCon(img)
        if maxdata[0] < 500:
            raise ValueError
        imgWarp = cv.resize(imgWarp, (125, 125))
        cv.imshow('imgWarp', imgWarp)

        imgWarp = imgWarp.astype(np.float32)
        imgWarp = imgWarp / 255
        imgWarp = imgWarp.reshape((1, 125, 125, 3))
        predict = np.argmax(model.predict(imgWarp)) + 1
        txt = get_card_data(predict)
        # 在攝影機畫面上寫上預測結果
        img = Image.fromarray(img)
        drawObj = ImageDraw.Draw(img)
        drawObj.text((220, 50), txt, fill='blue', font=fontInfo)
        img = np.array(img)
    except:
        imgWarp = np.array([])

    cv.imshow('img', img)
    
    ###############存檔成影片###########
    rowMV, colMV = img.shape[:2]
    # 影像上下顛倒 再者旋轉180度 
    M = cv.getRotationMatrix2D(((colMV - 1) / 2.0, (rowMV - 1) / 2.0), 180, 1)
    img = cv.warpAffine(img, M, (colMV, rowMV))
    # 影像左右相反 在這裡調整
    img = img[:, ::-1, :]
    img = cv.flip(img, 0)
    # 儲存影片
    out.write(img)
    ###############################
    k = cv.waitKey(50)
    if k == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()
