#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PyQt5 import QtCore, QtGui, QtWidgets


class MainLayout(object):

    def setupUi(self,window):
        window.resize(1000, 600)
        #窗口标题
        self.setWindowTitle('数字图像处理')

        self.centralWidget = QtWidgets.QWidget(window)      

        #全局布局
        mainLayout=QVBoxLayout(self.centralWidget)
        self.label = QLabel()
        self.show()

        #顶部布局
        #顶部固定布局
        topLayout=QHBoxLayout()
        solidLayout=QHBoxLayout()
        self.importImageEdit=QLineEdit()
        self.importImageEdit.setFocusPolicy(Qt.NoFocus)
        solidLayout.addWidget(self.importImageEdit)
        #文件按钮
        self.importButton=QPushButton('文件')
        solidLayout.addWidget(self.importButton)
        filemenu = QMenu(self)
        self.openAct = QAction('打开',self)
        filemenu.addAction(self.openAct)
        self.saveAct = QAction('保存',self)
        filemenu.addAction(self.saveAct)
        self.exitAct = QAction('退出',self)
        filemenu.addAction(self.exitAct)
        self.importButton.setMenu(filemenu)

        #灰度变换按钮
        self.GrayButton=QPushButton('灰度变换')
        solidLayout.addWidget(self.GrayButton)
        graymenu = QMenu(self)
        self.expAct = QAction('指数灰度变换',self)
        graymenu.addAction(self.expAct)
        self.reverseAct = QAction('负片',self)
        graymenu.addAction(self.reverseAct)
        self.GrayButton.setMenu(graymenu)

        #伽马矫正按钮
        self.GammaButton=QPushButton('伽马矫正')
        solidLayout.addWidget(self.GammaButton)

        #滤波按钮
        self.FilterButton=QPushButton('滤波')
        solidLayout.addWidget(self.FilterButton)
        filtermenu = QMenu(self)
        self.avgFilter = QAction('均值滤波',self)
        filtermenu.addAction(self.avgFilter)
        self.medFilter = QAction('中值滤波',self)
        filtermenu.addAction(self.medFilter)
        self.FilterButton.setMenu(filtermenu)

        #拉普拉斯锐化按钮
        self.LaplaceButton=QPushButton('拉普拉斯锐化')
        solidLayout.addWidget(self.LaplaceButton)

        #傅里叶变换按钮
        self.FourierButton=QPushButton('傅里叶变换')
        solidLayout.addWidget(self.FourierButton)

        #直方图均衡化按钮
        self.HistogramButton=QPushButton('直方图均衡化')
        solidLayout.addWidget(self.HistogramButton)

        #频率域滤波按钮
        self.FrequencyButton=QPushButton('频率域滤波')
        solidLayout.addWidget(self.FrequencyButton)
        frequencymenu = QMenu(self)
        self.ButterworthHighAct = QAction('布特沃思高通滤波',self)
        frequencymenu.addAction(self.ButterworthHighAct)
        self.ButterworthLowAct = QAction('布特沃思低通滤波',self)
        frequencymenu.addAction(self.ButterworthLowAct)
        self.FrequencyButton.setMenu(frequencymenu)

        #图像复原按钮
        self.RestoreButton=QPushButton('图像复原')
        solidLayout.addWidget(self.RestoreButton)
        restoremenu = QMenu(self)
        self.InverseFilteringAct = QAction('频率域逆滤波',self)
        restoremenu.addAction(self.InverseFilteringAct)
        self.WienerFilteringAct = QAction('维纳滤波',self)
        restoremenu.addAction(self.WienerFilteringAct)
        self.RestoreButton.setMenu(restoremenu)
        solidLayout.addStretch(1)
        topLayout.addLayout(solidLayout)

        #顶部隐藏布局
        self.hideLayout=QHBoxLayout()
        topLayout.addLayout(self.hideLayout)
        mainLayout.addLayout(topLayout)

        #中间布局
        midLayout=QHBoxLayout()
        self.showImageView=QTableWidget()
        midLayout.addWidget(self.showImageView)
        mainLayout.addLayout(midLayout)

        #底部布局
        bottomLayout=QHBoxLayout()
        self.preButton=QPushButton('上一张')
        bottomLayout.addWidget(self.preButton)
        self.nextButton=QPushButton('下一张')
        bottomLayout.addWidget(self.nextButton)
        bottomLayout.addStretch(4)
        self.exitButton=QPushButton('退出')
        bottomLayout.addWidget(self.exitButton)
        mainLayout.addLayout(bottomLayout)

        #设置stretch
        mainLayout.setStretchFactor(topLayout,1)
        mainLayout.setStretchFactor(midLayout,6)
        mainLayout.setStretchFactor(bottomLayout,1)

        window.setCentralWidget(self.centralWidget)



