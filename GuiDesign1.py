#-*- coding:utf-8 -*-

import image_func
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import cv2
import copy
import numpy as np
import os



WINDOW_WIDTH = 720
WINDOW_HEIGHT = 576
BACKGROUND_LOW = np.array([0,0,221])
BACKGROUND_HIGH = np.array([180,30,255])

#define new elements based on QLabel
# class llabel(QLabel):
#     def __init__(self):
#         super(llabel, self).__init__()
#         self.m_pixmap = None
#
#     def setImage(self, image):
#         self.m_pixmap = image
#
#     def paintEvent(self, event):
#         super().paintEvent(event)
#         if self.m_pixmap is None:
#             return
#         scale_jpg = self.m_pixmap.scaled(self.size(), Qt.KeepAspectRatio|Qt.SmoothTransformation)
#         self.setPixmap(scale_jpg)

class llabel(QLabel):
    signal_selected_hsv = pyqtSignal(tuple)
    def __init__(self):
        super(llabel, self).__init__()
        self.m_pixmap = None
        self.pos_x = None
        self.pos_y = None

    def setImage(self, image):
        self.m_pixmap = image

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.m_pixmap is None:
            return
        scale_jpg = self.m_pixmap.scaled(self.size(), Qt.KeepAspectRatio | Qt.SmoothTransformation)
        self.setPixmap(scale_jpg)

    def mousePressEvent(self, event):
        pos_x = event.pos().x()
        pos_y = event.pos().y()
        #grab  windows
        screen = QGuiApplication.primaryScreen()
        qmap = screen.grabWindow(self.winId(),pos_x,pos_y,1,1)
        qpixels = qmap.toImage()
        c = qpixels.pixel(0,0)
        hsv = QColor(c).getHsvF()
        h = int(hsv[0]*180)
        s = int(hsv[1]*256)
        v = int(hsv[2]*256)
        print(h,s,v,pos_x,pos_y)

    def mouseDoubleClickEvent(self, event):
        pos_x = event.pos().x()
        pos_y = event.pos().y()
        # grab  windows
        screen = QGuiApplication.primaryScreen()
        qmap = screen.grabWindow(self.winId(), pos_x, pos_y, 1, 1)
        qpixels = qmap.toImage()
        c = qpixels.pixel(0, 0)
        hsv = QColor(c).getHsvF()
        self.signal_selected_hsv.emit(hsv)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setWindowTitle("图像处理程序 版本： 0.8")
        self.resize(WINDOW_WIDTH,WINDOW_HEIGHT)
        self.stitched_image = None
        self.img_contour = None
        self.img_grid = None
        self.contour_list = None
        self.color_count = 0
        self.select_h = 0
        self.select_s = 0
        self.select_v = 0

        #provide font setting
        self.font_level0 = QFont()
        self.font_level0.setFamily("宋体")
        self.font_level0.setPointSizeF(10)
        self.font_level0.setBold(True)

        self.font_level1 = QFont()
        self.font_level1.setFamily("宋体")
        self.font_level1.setPointSizeF(8)
        self.font_level1.setBold(False)

        self.initUI()

    def initUI(self):
        wwg  = QWidget()
        hwg1 = QFrame()
        hwg1.setFrameShape(QFrame.StyledPanel)
        hwg2 = QFrame()
        hwg2.setFrameShape(QFrame.StyledPanel)
        hwg3 = QFrame()
        hwg3.setFrameShape(QFrame.StyledPanel)

        #UI design for main function region
        hbox = QHBoxLayout()
        hsplitter = QSplitter(Qt.Horizontal)
        vbox = QGridLayout()
        vbox.setAlignment(Qt.AlignTop)
        label0 = QLabel('主功能区')
        label0.setAlignment(Qt.AlignCenter)
        label0.setFont(self.font_level0)
        vbox.addWidget(label0,0,0)
        butn0 = QPushButton('图像拼接')
        butn1 = QPushButton('提取图像轮廓')
        butn2 = QPushButton('HSV 3D直方图统计')
        butn3 = QPushButton('图像网格化')
        butn4 = QPushButton('组分图像生成')

        butn0.setFont(self.font_level1)
        butn1.setFont(self.font_level1)
        butn2.setFont(self.font_level1)
        butn3.setFont(self.font_level1)
        butn4.setFont(self.font_level1)

        vbox.addWidget(butn0,1,0)
        vbox.addWidget(butn1,2,0)
        vbox.addWidget(butn2,4,0)
        vbox.addWidget(butn3,3,0)
        vbox.addWidget(butn4,5,0)
        hwg1.setLayout(vbox)

        #UI design for information region
        self.vbox1 = QGridLayout()
        self.vbox1.setAlignment(Qt.AlignTop)
        label1 = QLabel('信息显示区')
        label1.setFont(self.font_level0)
        label1.setAlignment(Qt.AlignCenter)
        self.vbox1.addWidget(label1,0,0)
        self.label2 = QLabel('准备接收命令...')
        self.label2.setAlignment(Qt.AlignTop|Qt.AlignHCenter)
        self.vbox1.addWidget(self.label2,1,0)
        self.vbox1.setRowStretch(0,20)
        self.vbox1.setRowStretch(1,80)
        hwg2.setLayout(self.vbox1)

        #layout for the first UI row
        hsplitter.addWidget(hwg1)
        hsplitter.addWidget(hwg2)
        hsplitter.setStretchFactor(0,50)
        hsplitter.setStretchFactor(1,50)
        hsplitter.setAutoFillBackground(True)

        #image show region by QLabel
        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(hsplitter)
        vbox2 = QVBoxLayout()
        hwg3.setLayout(vbox2)
        vbox2.setAlignment(Qt.AlignTop)
        label3 = QLabel('图片显示区')
        label3.setFont(self.font_level0)
        label3.setAlignment(Qt.AlignTop|Qt.AlignHCenter)
        vbox2.addWidget(label3)
        self.label4 = llabel()
        self.label4.setAlignment(Qt.AlignCenter)
        self.setLabel4_backgroud()
        vbox2.addWidget(self.label4)
        vbox2.setStretch(0,5)
        vbox2.setStretch(1,95)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(hwg3)
        vsplitter.setStretchFactor(1,4)
        vsplitter.setStretchFactor(2,6)
        vsplitter.setAutoFillBackground(True)


        hbox.addWidget(vsplitter)
        wwg.setLayout(hbox)
        self.setCentralWidget(wwg)

        #bind the event to slot function
        butn0.clicked.connect(self.OnClickButton_stitch)
        butn1.clicked.connect(self.OnClickButton_contour)
        butn2.clicked.connect(self.OnClickButton_histogram)
        butn3.clicked.connect(self.OnClickButton_grid)
        butn4.clicked.connect(self.OnClickButton_mask)


    def OnClickButton_stitch(self):
        self.dynamic_widget_buttn0()


    def setLabel4_backgroud(self):
        pe = QPalette()
        pe.setColor(QPalette.Background,Qt.black)
        self.label4.setAutoFillBackground(True)
        self.label4.setPalette(pe)

    def OnClickButton_contour(self):
        if self.stitched_image is None:
            self.label2.setText('没有拼接的图像，无法进行下一步计算，请先运行图像拼接命令！')
            self.label2.repaint()
            return

        self.label2.setText('开始提取图像轮廓，请等待....')
        self.label2.repaint()
        self.img_contour = copy.deepcopy(self.stitched_image)
        cx, cy, c, max_x, max_y, min_x, min_y, self.img_contour = image_func.get_contour(self.img_contour)
        if cx is None:
            self.label2.setText('提取失败，请检查拼接图像是否正常！')
            self.label2.repaint()
            return
        #store the contour information
        self.contour_list = (cx,cy,c,max_x,max_y,min_x,min_y)
        img_out = self.img_contour
        img1 = cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)
        QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.label4.setImage(pixmap)
        self.label4.repaint()
        self.label2.setText('提取成功！')
        self.label2.repaint()

    def OnClickButton_histogram(self):
        if self.stitched_image is None:
            self.label2.setText('没有拼接的图像，无法进行下一步计算，请先运行图像拼接命令！')
            self.label2.repaint()
            return

        self.label2.setText('开始计算颜色直方图(色域：HSV)，请等待....')
        self.label2.repaint()
        image_func.get_histogram_3d(self.stitched_image)
        self.label2.setText('开始显示!')
        self.label2.repaint()


    def OnClickButton_grid(self):
        self.dynamic_widget_button3()


    def dynamic_widget_buttn0(self):
        vbox = QGridLayout()
        self.widget_button0 = QWidget()
        self.widget_button0.setWindowTitle('拼接图像输入信息框')
        label00 = QLabel("待拼接图像输入目录:")
        self.edit0 = QLineEdit()
        self.label01 = QPushButton('...')
        self.label01.clicked.connect(self.OnClickButtonFile0)
        label10 = QLabel("拼接图像输出目录:")
        self.edit1 = QLineEdit()
        self.label11 = QPushButton('...')
        self.label11.clicked.connect(self.OnClickButtonFile1)
        self.label21 = QPushButton("开始拼接")
        self.label21.clicked.connect(self.OnClickStitchStart)
        vbox.addWidget(label00,0,0)
        vbox.addWidget(self.edit0,0,1)
        vbox.addWidget(self.label01,0,2)
        vbox.addWidget(label10,1,0)
        vbox.addWidget(self.edit1,1,1)
        vbox.addWidget(self.label11,1,2)
        vbox.addWidget(self.label21,2,1)
        self.widget_button0.setLayout(vbox)
        self.widget_button0.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.widget_button0.move(200,200)
        self.widget_button0.show()

    def OnClickButtonFile0(self):
        dir = QFileDialog.getExistingDirectory(self, '选择输入文件夹', './')
        self.edit0.setText(dir)


    def OnClickButtonFile1(self):
        dir = QFileDialog.getExistingDirectory(self, '选择输出文件夹', './')
        self.edit1.setText(dir)

    def OnClickStitchStart(self):
        self.widget_button0.close()
        self.label2.setText('开始拼接图像，请等待.....')
        self.label2.repaint()
        stitch_dir = self.edit0.text()
        filenames = image_func.scan_directory(stitch_dir)
        images = image_func.load_images(filenames)
        self.stitched_image = image_func.image_stitch(images)
        if self.stitched_image is None:
            self.label2.setText('拼接失败，请检查拍摄的输入图像，需要保证至少25%的重合拍摄!')
            self.label2.repaint()
        else:
            # write stitched image picture to output dir
            stitched_filename = os.path.join(self.edit1.text(), './stitched.jpg')
            cv2.imwrite(stitched_filename, self.stitched_image)

            self.label2.setText('拼接成功!')
            self.label2.repaint()
            # display the image
            img1 = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2RGB)
            QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            self.label4.setImage(pixmap)
            self.label4.repaint()

    def dynamic_widget_button3(self):
        vbox = QGridLayout()
        self.widget_button3 = QWidget()
        self.widget_button3.setWindowTitle('网格化行列设置')
        label0 = QLabel('网格行数设置:')
        self.button3_edit0 = QLineEdit()
        label1 = QLabel('网格列数设置:')
        self.button3_edit1 = QLineEdit()
        self.button3_start = QPushButton('开始')
        self.button3_start.clicked.connect(self.OnClickedButton3Start)
        vbox.addWidget(label0,0,0)
        vbox.addWidget(self.button3_edit0,0,1)
        vbox.addWidget(label1,1,0)
        vbox.addWidget(self.button3_edit1,1,1)
        vbox.addWidget(self.button3_start,2,0)
        self.widget_button3.setLayout(vbox)
        self.widget_button3.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.widget_button3.move(100, 100)
        self.widget_button3.show()

    def OnClickedButton3Start(self):
        self.widget_button3.close()

        if self.contour_list is None:
            text = '网格化失败，请先运行提取轮廓命令！'
            self.label2.setText(text)
            self.label2.repaint()
            return

        self.label2.repaint()
        row = self.button3_edit0.text()
        col = self.button3_edit1.text()
        text = '网格行数：'+row+',网格列数:'+col+'\n'+'开始网格化！'
        self.label2.setText(text)
        (cx, cy, c,max_x, max_y, min_x, min_y) = self.contour_list
        self.img_grid = copy.deepcopy(self.img_contour)
        self.img_grid = image_func.draw_grid(self.img_grid,cx,cy,max_x,max_y,min_x,min_y,int(row),int(col))

        #write grid image picture to output dir
        grid_filename = os.path.join(self.edit1.text(),'./grid_image.jpg')
        cv2.imwrite(grid_filename,self.img_grid)

        img1 = cv2.cvtColor(self.img_grid, cv2.COLOR_BGR2RGB)
        QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.label4.setImage(pixmap)
        self.label4.repaint()
        text = '网格行数：' + row + ',网格列数:' + col + '\n' + '显示网格！\n（注意：不支持分数像素，因此划分的网格可能不均匀）'
        self.label2.setText(text)


    def OnClickButton_mask(self):
        self.dynamic_widget_button4()


    def dynamic_widget_button4(self):
        vbox = QGridLayout()
        self.widget_button4 = QWidget()
        self.widget_button4.setWindowTitle('选择HSV颜色区间')
        label0 = QLabel('(颜色通道H) from:')
        self.button4_edit0 = QLineEdit()
        label1= QLabel('to:')
        self.button4_edit1 = QLineEdit()
        checkbox1 = QCheckBox("辅助模式")
        label2 = QLabel('(颜色通道S) from：')
        self.button4_edit2 = QLineEdit()
        label3 = QLabel('to:')
        self.button4_edit3 = QLineEdit()
        label4 = QLabel('(颜色通道V) from:')
        self.button4_edit4 = QLineEdit()
        label5 = QLabel('to:')
        self.button4_edit5 = QLineEdit()

        label6 = QLabel('网格设置 行数：')
        self.button4_edit6 = QLineEdit()
        label7 = QLabel('列数:')
        self.button4_edit7 = QLineEdit()

        self.checkbox0 = QCheckBox('去除')
        self.checkbox1 = QCheckBox('组分图像背景设为白色')

        vbox.addWidget(label0,0,0)
        vbox.addWidget(self.button4_edit0,0,1)
        vbox.addWidget(label1,0,2)
        vbox.addWidget(self.button4_edit1,0,3)
        vbox.addWidget(checkbox1,0,4)

        vbox.addWidget(label2, 1, 0)
        vbox.addWidget(self.button4_edit2, 1, 1)
        vbox.addWidget(label3, 1, 2)
        vbox.addWidget(self.button4_edit3, 1, 3)

        vbox.addWidget(label4, 2, 0)
        vbox.addWidget(self.button4_edit4, 2, 1)
        vbox.addWidget(label5, 2, 2)
        vbox.addWidget(self.button4_edit5, 2, 3)

        vbox.addWidget(label6, 3, 0)
        vbox.addWidget(self.button4_edit6, 3, 1)
        vbox.addWidget(label7, 3, 2)
        vbox.addWidget(self.button4_edit7, 3, 3)

        vbox.addWidget(self.checkbox0,4,0)
        vbox.addWidget(self.checkbox1, 4, 1)
        self.button4_start = QPushButton('开始')
        vbox.addWidget(self.button4_start,4,2)

        checkbox1.stateChanged.connect(self.OnClickCheckbox)
        self.button4_start.clicked.connect(self.OnClickedButton4Start)
        self.widget_button4.setLayout(vbox)
        self.widget_button4.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.widget_button4.move(100, 100)
        self.widget_button4.show()

    def OnClickCheckbox(self):
        self.dynamic_widget_button5()

    def receive_selected_color(self,hsv):
        h = int(hsv[0] * 180)
        s = int(hsv[1] * 256)
        v = int(hsv[2] * 256)

        if self.color_count == 0:
            self.select_h = h
            self.select_s = s
            self.select_v = v
            self.label2.setText('已经选择了一个像素，其HSV值为(%d,%d,%d)，请继续选择第二个像素......'% (h,s,v))
            self.label2.repaint()
            self.color_count = 1
            return
        else :
            self.label2.setText('两个像素选择完成，第二个像素的HSV值为(%d,%d,%d)' % (h,s,v))
            self.label2.repaint()
            delta = 0
            h_range_low = min(h-delta,self.select_h-delta)
            h_range_high = max(h+delta,self.select_h+delta)
            s_range_low = min(s - delta, self.select_s - delta)
            s_range_high = max(s + delta, self.select_s + delta)
            v_range_low = min(h - delta, self.select_v - delta)
            v_range_high = max(h + delta, self.select_v + delta)

            h_range_low = max(0,h_range_low)
            h_range_high = min(180, h_range_high)
            s_range_low = max(0,s_range_low)
            s_range_high = min(255, s_range_high)
            v_range_low = max(0,v_range_low)
            v_range_high = min(255, v_range_high)
            self.color_count = 0

        # if h >= range_purple_h[0] and h <= range_purple_h[1]:
        #     h_range_low = range_purple_h[0]
        #     h_range_high = range_purple_h[1]
        # elif h >= range_red_h0[0] and h <= range_red_h0[1]:
        #     h_range_low = range_red_h0[0]
        #     h_range_high = range_red_h0[1]
        # elif h >= range_red_h1[0] and h <= range_red_h1[1]:
        #     h_range_low = range_red_h1[0]
        #     h_range_high = range_red_h1[1]

        self.button4_edit0.setText(str(h_range_low))
        self.button4_edit1.setText(str(h_range_high))
        self.button4_edit2.setText(str(s_range_low))
        self.button4_edit3.setText(str(s_range_high))
        self.button4_edit4.setText(str(v_range_low))
        self.button4_edit5.setText(str(v_range_high))



    def dynamic_widget_button5(self):
        text = '请在图像中双击选择两个像素来设置图像的HSV范围......'
        self.label2.setText(text)
        self.label2.repaint()

        self.widget_checkbox = QWidget()
        vbox = QGridLayout()
        self.widget_checkbox.setWindowTitle("颜色选择")
        label1_checkbox = llabel()
        label1_checkbox.signal_selected_hsv.connect(self.receive_selected_color)
        img1 = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2RGB)
        QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        label1_checkbox.setImage(pixmap)
        vbox.addWidget(label1_checkbox)
        self.widget_checkbox.setLayout(vbox)
        self.widget_checkbox.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.widget_checkbox.setGeometry(100,100,740,580)
        self.widget_checkbox.show()


    def OnClickedButton4Start(self):
        self.widget_checkbox.close()
        self.widget_button4.close()

        if self.stitched_image is None:
            text = '生成组分图像失败，请先运行图像拼接命令！'
            self.label2.setText(text)
            self.label2.repaint()
            return

        if self.contour_list is None:
            text = '生成组分图像失败，请先运行提取轮廓命令！'
            self.label2.setText(text)
            self.label2.repaint()
            return

        text = '开始生成组分图像并且抽取统计信息，请等待......'+'\n' + '（注意：不支持分数像素，因此划分的网格可能不均匀）'
        self.label2.setText(text)
        self.label2.repaint()

        h_low = int(self.button4_edit0.text())
        h_high = int(self.button4_edit1.text())
        s_low = int(self.button4_edit2.text())
        s_high = int(self.button4_edit3.text())
        v_low = int(self.button4_edit4.text())
        v_high = int(self.button4_edit5.text())
        row  =  int(self.button4_edit6.text())
        col  =  int(self.button4_edit7.text())
        remove_enable = int(self.checkbox0.isChecked())
        white_flag = int(self.checkbox1.isChecked())

        low_range = np.array([h_low,s_low,v_low])
        high_range = np.array([h_high,s_high,v_high])
        (cx, cy, c,max_x, max_y, min_x, min_y) = self.contour_list
        out_dir = self.edit1.text()
        mask_hsv,img_hsv = image_func.get_color_mask_image(self.stitched_image,low_range,high_range)
        mask_hsv = image_func.generate_final_mask(img_hsv,mask_hsv,c,remove_enable,BACKGROUND_LOW,BACKGROUND_HIGH)
        grid_row, grid_col = image_func.get_grid_info(mask_hsv, cx, cy, max_x, max_y, min_x, min_y, row, col)
        image_func.get_statistics_per_bin(mask_hsv,grid_row,grid_col,out_dir)
        res = image_func.generate_image_from_mask(self.stitched_image,mask_hsv,cx,cy,c,max_x,max_y,min_x,min_y,row,col,white_flag)

        # write masked image picture to output dir
        masked_filename = os.path.join(out_dir,'./masked_image.jpg')
        cv2.imwrite(masked_filename, res)

        #show origin and new image
        res = np.concatenate((self.stitched_image,res),axis=1)


        #keep net grid
        img1 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.label4.setImage(pixmap)
        self.label4.repaint()
        text = '成功完成组分图像的处理，统计信息已经存储到excel文件中！'
        self.label2.setText(text)
        self.label2.repaint()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

