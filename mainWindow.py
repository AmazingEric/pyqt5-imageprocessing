#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from mainWindowLayout import MainLayout

import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from math import sqrt, pow, exp

class MainWindow(QMainWindow, MainLayout):
    imagePaths = []
    originImages = [] 
    imageList = []  # 二维的图像列表，用于存储原图像以及处理后的图像
    hideLayoutTag = -1

    def __init__(self,parent = None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.signalSlots()
        setHideButton(self)
        
    def keyPressEvent(self, event):
        #使用键盘输入左右键来实现图像切换
        if (event.key() == Qt.Key_Left or event.key()==Qt.Key_Up):
            preImage(self)
            print('上一张')
        if (event.key() == Qt.Key_Right or event.key()==Qt.Key_Down):
            nextImage(self)
            print('下一张')

    # 绑定按钮与具体方法
    def signalSlots(self):
        # 文件按钮相关方法
        # 打开
        self.openAct.triggered.connect(lambda : importImage(self))
        # 保存
        self.saveAct.triggered.connect(lambda : importImage(self))
        # 退出
        self.exitAct.triggered.connect(self.close)

        # 灰度变换按钮相关方法
        # 指数灰度变换
        self.expAct.triggered.connect(lambda : ExpGray(self))
        # 负片
        self.reverseAct.triggered.connect(lambda : ReverseGray(self))

        # 伽马矫正方法
        self.GammaButton.clicked.connect(lambda : GammaChange(self))

        # 滤波按钮相关方法
        # 均值滤波
        self.avgFilter.triggered.connect(lambda : AvgFilter(self))
        # 中值滤波
        self.medFilter.triggered.connect(lambda : MedFilter(self))

        # 拉普拉斯锐化方法
        self.LaplaceButton.clicked.connect(lambda : Laplacian(self))

        # 傅里叶变换方法
        self.FourierButton.clicked.connect(lambda : FourierChange(self))

        # 直方图均衡化方法
        self.HistogramButton.clicked.connect(lambda : HistogramEqualization(self))

        # 频率域滤波按钮相关方法
        # 布特沃思高通滤波
        self.ButterworthHighAct.triggered.connect(lambda : ButterworthHigh(self))
        # 布特沃思低通滤波
        self.ButterworthLowAct.triggered.connect(lambda : ButterworthLow(self))

        # 图像复原相关方法
        # 频率域逆滤波
        self.InverseFilteringAct.triggered.connect(lambda : InverseFiltering(self))
        # 维纳滤波
        self.WienerFilteringAct.triggered.connect(lambda : WienerFiltering(self))

        # 底部按钮
        # 上一张
        self.preButton.clicked.connect(lambda : preImage(self))
        # 下一张
        self.nextButton.clicked.connect(lambda : nextImage(self))
        # 退出
        self.exitButton.clicked.connect(self.close)


# 灰度变换按钮相关方法
# 指数灰度变换
def ExpGray(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        img_info = img[0].shape
        img_height = img_info[0]
        img_width = img_info[1]
        result = []
        result = np.arange(img_height*img_width*3, dtype='uint8').reshape(img_height, img_width, 3)
        for i in range(img_height):
            for j in range(img_width):
                for k in range(3):
                    # 对原图像的像素进行指数处理
                    result[i][j][k] = 30.0 * np.log2(img[0][i][j][k]+1)
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','指数灰度变换'])

# 负片
def ReverseGray(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 对原图像进行负片处理
        result = 255-img[0]
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','负片'])


# 伽马矫正按钮方法
def GammaChange(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        img_info = img[0].shape
        img_height = img_info[0]
        img_width = img_info[1]
        result = []
        result = np.arange(img_height*img_width*3, dtype='uint8').reshape(img_height, img_width, 3)
        for i in range(img_height):
            for j in range(img_width):
                for k in range(3):
                    # 对原图像的像素进行伽马矫正
                    result[i][j][k] = 1.0 * np.power(img[0][i][j][k]/255.0, 1/0.6) * 255.0
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','伽马矫正'])

#滤波按钮相关方法
#均值滤波
def AvgFilter(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 对原图像进行均值滤波
        result = cv2.blur(img[0], (5, 5))
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','均值滤波'])

#中值滤波
def MedFilter(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 对原图像进行中值滤波
        result = cv2.medianBlur(img[0],5)
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','中值滤波'])


#拉普拉斯锐化按钮方法
def Laplacian(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        img_height, img_width = img[0].shape[:2]
        # 获得原图像的拉普拉斯算子
        laplace = cv2.Laplacian(img[0], cv2.CV_64F, ksize=3)
        laplace[laplace<0] = 0
        laplace[laplace>255] = 255
        # 原图像减去拉普拉斯算子，得到锐化后的图像
        result = img[0] - laplace
        result[result<0] = 0
        result[result>255] = 255
        result = cv2.resize(src=result, dsize=(img_height, img_width)).astype('uint8')
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','拉普拉斯锐化'])


# 傅里叶变换按钮方法
def FourierChange(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 提取图像的三个通道
        b,g,r = cv2.split(img[0])
        # 对每个通道单独进行傅里叶变换
        b_freImg,b_recImg = oneChannelDft(b)
        g_freImg, g_recImg = oneChannelDft(g)
        r_freImg, r_recImg = oneChannelDft(r)
        # 将处理后的通道合并成一个图像
        freImg = cv2.merge([b_freImg,g_freImg,r_freImg])
        imgs.extend([img[0],freImg])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','傅里叶变换'])
def oneChannelDft(img):
    width, height = img.shape
    nwidth = cv2.getOptimalDFTSize(width)
    nheigth = cv2.getOptimalDFTSize(height)
    nimg = np.zeros((nwidth, nheigth))
    nimg[:width, :height] = img
    dft = cv2.dft(np.float32(nimg), flags = cv2.DFT_COMPLEX_OUTPUT)
    ndft = dft[:width, :height]
    ndshift = np.fft.fftshift(ndft)
    magnitude = np.log(cv2.magnitude(ndshift[:, :, 0], ndshift[:, :, 1]))
    result = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255
    frequencyImg = result.astype('uint8')
    ilmg = cv2.idft(dft)
    ilmg = cv2.magnitude(ilmg[:, :, 0], ilmg[:, :, 1])[:width, :height]
    ilmg = np.floor((ilmg - ilmg.min()) / (ilmg.max() - ilmg.min()) * 255)
    recoveredImg = ilmg.astype('uint8')
    return frequencyImg,recoveredImg

# 直方图均衡化按钮方法
def HistogramEqualization(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        b, g, r = cv2.split(img[0])
        b_equal = cv2.equalizeHist(b)
        g_equal = cv2.equalizeHist(g)
        r_equal = cv2.equalizeHist(r)
        result = cv2.merge([b_equal, g_equal, r_equal])
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','直方图均衡化'])

# 频率域滤波按钮相关方法
# 布特沃斯高通滤波
def ButterworthHigh(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 提取图像的三个通道
        B, G, R = cv2.split(img[0])
        # 对每个通道单独进行布特沃斯高通滤波，D0取20
        B = OneChannelButterworth(B, 1, 20)
        G = OneChannelButterworth(G, 1, 20)
        R = OneChannelButterworth(R, 1, 20)
        # 将处理后的通道合并成一个图像
        result = cv2.merge([B, G, R])
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','布特沃斯高通滤波'])

# 布特沃斯低通滤波
def ButterworthLow(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 提取图像的三个通道
        B, G, R = cv2.split(img[0])
        # 对每个通道单独进行布特沃斯低通滤波，D0取80
        B = OneChannelButterworth(B, 0, 80)
        G = OneChannelButterworth(G, 0, 80)
        R = OneChannelButterworth(R, 0, 80)
        # 将处理后的通道合并成一个图像
        result = cv2.merge([B, G, R])
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','布特沃斯低通滤波'])

def OneChannelButterworth(image, method, D0):
    n = 2
    img_height, img_width = image.shape[:2]
    dft_img = np.fft.fft2(image)
    dft_img = np.fft.fftshift(dft_img)
    H = np.zeros_like(dft_img)
    for i in range(img_height):
        for j in range(img_width):
            D = sqrt((pow(i-img_height/2, 2)+pow(j-img_width/2, 2)))
            H[i][j] = 1./(1+pow(D/D0, 2*n))
    if method:
        result = dft_img*(1-H)
    else:
        result = dft_img*H
    idft_img = np.fft.ifftshift(result)
    idft_img = np.fft.ifft2(idft_img)
    result = np.abs(np.real(idft_img))
    result = np.clip(result,0,255)
    return result.astype('uint8')


# 图像复原按钮相关方法
# 频率逆滤波
def InverseFiltering(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 提取图像的三个通道
        B, G, R = cv2.split(img[0])
        # 对每个通道单独进行频率逆滤波
        B = OneChannelIF(B)
        G = OneChannelIF(G)
        R = OneChannelIF(R)
        # 将处理后的通道合并成一个图像
        result = cv2.merge([B, G, R])
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','频率域逆滤波图像复原'])
def OneChannelIF(image):
    img_height, img_width = image.shape[:2]
    result = np.zeros_like(image, dtype=complex)
    # 先对原图像进行傅里叶变换并移至中心
    dft_img = np.fft.fft2(image)
    dft_img = np.fft.fftshift(dft_img)
    for i in range(img_height):
        for j in range(img_width):
            result[i][j] = dft_img[i][j] / H(i,j)
    # 对图像进行逆傅里叶变换
    idft_img = np.fft.ifftshift(result)
    idft_img = np.fft.ifft2(idft_img)
    result = np.abs(np.real(idft_img))
    result = np.clip(result,0,255)
    return result.astype('uint8')

#维纳滤波
def WienerFiltering(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        # 提取图像的三个通道
        B, G, R = cv2.split(img[0])
        # 对每个通道单独进行维纳滤波
        B = OneChannelWF(B)
        G = OneChannelWF(G)
        R = OneChannelWF(R)
        # 将处理后的通道合并成一个图像
        result = cv2.merge([B, G, R])
        imgs.extend([img[0],result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window,['原图','维纳滤波图像复原'])
def OneChannelWF(image):
    K = 0.0001
    img_height, img_width = image.shape[:2]
    result = np.zeros_like(image, dtype=complex)
    # 先对原图像进行傅里叶变换并移至中心
    dft_img = np.fft.fft2(image)
    dft_img = np.fft.fftshift(dft_img)
    for i in range(img_height):
        for j in range(img_width):
            result[i][j] = (H(i,j)/(pow(H(i,j),2)+K))*dft_img[i][j]
    # 对图像进行逆傅里叶变换
    idft_img = np.fft.ifftshift(result)
    idft_img = np.fft.ifft2(idft_img)
    result = np.abs(np.real(idft_img))
    result = np.clip(result,0,255)
    return result.astype('uint8')

def H(u, v):
    k = 0.00001
    return exp(-1*k*pow(pow(u,2)+pow(v,2), 5/6))

#打开图像
def importImage(window):
    fname, _ = QFileDialog.getOpenFileName(window, 'Open file', '.', 'Image Files(*.jpg *.bmp *.png *.jpeg *.rgb *.tif)')
    if fname != '':
        window.importImageEdit.setText(fname)
        window.imagePaths = []
        window.originImages = []
        window.imageList = []
        window.imagePaths.append(fname)
    if window.imagePaths != []:
        readIamge(window)
        resizeFromList(window, window.originImages)
        showImage(window)
        showButton(window)

def readIamge(window):
    window.originImages = []
    for path in window.imagePaths:
        imgs = []
        img = cv2.imdecode(np.fromfile(path, dtype = np.uint8), 1)
        imgs.append(img)
        window.originImages.append(imgs)

#显示图像
def showImage(window,headers = []):
    window.showImageView.clear()
    window.showImageView.setColumnCount(len(window.imageList[0]))
    window.showImageView.setRowCount(len(window.imageList))

    window.showImageView.setShowGrid(False)
    window.showImageView.setEditTriggers(QAbstractItemView.NoEditTriggers)
    window.showImageView.setHorizontalHeaderLabels(headers)
    for x in range(len(window.imageList[0])):
        for y in range(len(window.imageList)):
            imageView = QGraphicsView()
            imageView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            imageView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            img = window.imageList[y][x]
            width = img.shape[1]
            height = img.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            window.showImageView.setColumnWidth(x, width)
            window.showImageView.setRowHeight(y, height)

            frame = QImage(img, width, height, QImage.Format_RGB888)
            # 调用QPixmap命令，建立一个图像存放框
            pix = QPixmap.fromImage(frame)
            item = QGraphicsPixmapItem(pix) 
            scene = QGraphicsScene()  # 创建场景
            scene.addItem(item)
            imageView.setScene(scene)
            window.showImageView.setCellWidget(y, x, imageView)

def resizeFromList(window,imageList):
    width = 600
    height = 600
    window.imageList = []
    for x_pos in range(len(imageList)):
        imgs = []
        for img in imageList[x_pos]:
            image = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)

# 设置按钮不可见
def setHideButton(window):
    window.GrayButton.hide()
    window.GammaButton.hide()
    window.FilterButton.hide()
    window.LaplaceButton.hide()
    window.FourierButton.hide()
    window.HistogramButton.hide()
    window.FrequencyButton.hide()
    window.RestoreButton.hide()
    window.preButton.setEnabled(False)
    window.nextButton.setEnabled(False)

# 设置按钮可见
def showButton(window):
    window.GrayButton.setVisible(True)
    window.GammaButton.setVisible(True)
    window.FilterButton.setVisible(True)
    window.LaplaceButton.setVisible(True)
    window.FourierButton.setVisible(True)
    window.HistogramButton.setVisible(True)
    window.FrequencyButton.setVisible(True)
    window.RestoreButton.setVisible(True)
    window.preButton.setEnabled(True)
    window.nextButton.setEnabled(True)

# 获取所得文件的目录
def getDirFromFname(fname):
    pathList=fname.split('/')
    filename=pathList.pop()
    dirname='/'.join(pathList)
    return dirname,filename

# 上一页按钮事件
def preImage(window):
    fname=window.imagePaths[0]
    dirname,filename=getDirFromFname(fname)
    imageList=[]
    picTypes=['.jpg','.bmp','.png','.jpeg','.rgb','.tif']
    root, dirs, files = os.walk(dirname).__next__()
    for file in files:
        if(file[-4:] in picTypes or file[-5:] in picTypes):
            imageList.append(file)
    index=imageList.index(filename)
    if index==0:
        window.preButton.setEnabled(False)
        QMessageBox.information(window, "错误提示",  "这已经是第一张图片", QMessageBox.Ok)
    else: 
        window.nextButton.setEnabled(True)
        index=index-1
    fname=root+'/'+imageList[index]
    window.importImageEdit.setText(fname)
    window.imagePaths = []
    window.originImages = []
    window.imageList = []
    window.imagePaths.append(fname)
    if window.imagePaths != []:
        readIamge(window)
        resizeFromList(window, window.originImages)
        showImage(window)

# 下一页按钮事件
def nextImage(window):
    fname=window.imagePaths[0]
    dirname,filename=getDirFromFname(fname)
    imageList=[]
    picTypes=['.jpg','.bmp','.png','.jpeg','.rgb','.tif']
    root, dirs, files = os.walk(dirname).__next__()
    for file in files:
        if(file[-4:] in picTypes or file[-5:] in picTypes):
            imageList.append(file)
    index=imageList.index(filename)
    if index==len(imageList)-1:
        window.nextButton.setEnabled(False)
        QMessageBox.information(window, "错误提示",  "这已经是最后一张图片", QMessageBox.Ok)
    else: 
        window.preButton.setEnabled(True)
        index=index+1
    fname=root+'/'+imageList[index]
    window.importImageEdit.setText(fname)
    window.imagePaths = []
    window.originImages = []
    window.imageList = []
    window.imagePaths.append(fname)
    if window.imagePaths != []:
        readIamge(window)
        resizeFromList(window, window.originImages)
        showImage(window)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
