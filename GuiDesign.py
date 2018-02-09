import image_func
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import cv2
import copy


WINDOW_WIDTH = 720
WINDOW_HEIGHT = 576


class llabel(QLabel):
    def __init__(self):
        super(llabel, self).__init__()
        self.m_pixmap = None

    def setImage(self, image):
        self.m_pixmap = image

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.m_pixmap is None:
            return
        scale_jpg = self.m_pixmap.scaled(self.size(), Qt.KeepAspectRatio|Qt.SmoothTransformation)
        self.setPixmap(scale_jpg)



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setWindowTitle("图像处理程序 版本： 0.8")
        self.resize(WINDOW_WIDTH,WINDOW_HEIGHT)
        self.initUI()

    def initUI(self):
        wwg  = QWidget()
        hwg1 = QFrame()
        hwg1.setFrameShape(QFrame.StyledPanel)
        hwg2 = QFrame()
        hwg2.setFrameShape(QFrame.StyledPanel)
        hwg3 = QFrame()
        hwg3.setFrameShape(QFrame.StyledPanel)

        hbox = QHBoxLayout()
        hsplitter = QSplitter(Qt.Horizontal)
        vbox = QGridLayout()
        vbox.setAlignment(Qt.AlignTop)
        label0 = QLabel('主功能区')
        label0.setAlignment(Qt.AlignCenter)
        vbox.addWidget(label0,0,0)
        butn0 = QPushButton('图像拼接')
        butn1 = QPushButton('提取图像轮廓')
        butn2 = QPushButton('计算颜色统计图')
        butn3 = QPushButton('图像网格化')
        vbox.addWidget(butn0,1,0)
        vbox.addWidget(butn1,2,0)
        vbox.addWidget(butn2,3,0)
        vbox.addWidget(butn3,4,0)
        hwg1.setLayout(vbox)


        self.vbox1 = QGridLayout()
        self.vbox1.setAlignment(Qt.AlignTop)
        label1 = QLabel('信息显示区')
        label1.setAlignment(Qt.AlignCenter)
        self.vbox1.addWidget(label1,0,0)
        self.label2 = QLabel('准备接收命令...')
        self.label2.setAlignment(Qt.AlignTop|Qt.AlignHCenter)
        self.vbox1.addWidget(self.label2,1,0)
        self.vbox1.setRowStretch(0,20)
        self.vbox1.setRowStretch(1,80)
        hwg2.setLayout(self.vbox1)


        hsplitter.addWidget(hwg1)
        hsplitter.addWidget(hwg2)
        hsplitter.setStretchFactor(0,50)
        hsplitter.setStretchFactor(1,50)
        hsplitter.setAutoFillBackground(True)

        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(hsplitter)
        vbox2 = QVBoxLayout()
        hwg3.setLayout(vbox2)
        vbox2.setAlignment(Qt.AlignTop)
        label3 = QLabel('图片显示区')
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

        butn0.clicked.connect(self.OnClickButton0)
        butn1.clicked.connect(self.OnClickButton1)
        butn2.clicked.connect(self.OnClickButton2)
        butn3.clicked.connect(self.OnClickButton3)


    def OnClickButton0(self):
        self.dynamic_widget_buttn0()


    def setLabel4_backgroud(self):
        pe = QPalette()
        pe.setColor(QPalette.Background,Qt.black)
        self.label4.setAutoFillBackground(True)
        self.label4.setPalette(pe)

    def OnClickButton1(self):
        if self.stitched_image is None:
            self.label2.setText('图像拼接失败，无法进行下一步计算！')
            self.label2.repaint()
            return
        else:
            self.label2.setText('开始提取图像轮廓，请等待....')
            self.label2.repaint()
        cx, cy, c, max_x, max_y, min_x, min_y,self.img_contour = image_func.get_contour(self.stitched_image)
        if cx is None:
            self.label2.setText('提取失败，请检查拼接图像是否正常！')
            self.label2.repaint()
            return
        self.contour_list = (cx,cy,max_x,max_y,min_x,min_y)
        img_out = self.img_contour
        img1 = cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)
        QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.label4.setImage(pixmap)
        self.label4.repaint()
        self.label2.setText('提取成功！')
        self.label2.repaint()

    def OnClickButton2(self):
        self.label2.setText('开始计算颜色直方图(色域：HSV)，请等待....')
        self.label2.repaint()
        image_func.get_histogram(self.stitched_image)
        self.label2.setText('开始显示!')
        self.label2.repaint()


    def OnClickButton3(self):
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
        self.widget_button0.move(100,100)
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
        print(len(filenames))
        images = image_func.load_images(filenames)
        out_dir = self.edit1.text()
        self.stitched_image = image_func.image_stitch(images, out_dir)
        if self.stitched_image is None:
            self.label2.setText('拼接失败，请检查拍摄的输入图像，需要保证25%的重合拍摄!')
            self.label2.repaint()
        else:
            self.label2.setText('拼接成功!')
            self.label2.repaint()
            # display the image
            out_jpg = QPixmap(out_dir + '/stitched.jpg', )
            self.label4.setImage(out_jpg)
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

        self.label2.repaint()
        row = self.button3_edit0.text()
        col = self.button3_edit1.text()
        text = '网格行数：'+row+',网格列数:'+col+'\n'+'开始网格化！'
        self.label2.setText(text)
        (cx, cy, max_x, max_y, min_x, min_y) = self.contour_list
        img_copy = copy.deepcopy(self.img_contour)
        img_out = image_func.draw_grid(img_copy,cx,cy,max_x,max_y,min_x,min_y,int(row),int(col))
        img1 = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        QImg = QImage(img1.data, img1.shape[1], img1.shape[0], 3 * img1.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.label4.setImage(pixmap)
        self.label4.repaint()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

