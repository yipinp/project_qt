import image_func
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys


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
        scale_jpg = self.m_pixmap.scaled(self.size(), Qt.KeepAspectRatio)
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
        butn1 = QPushButton('计算重心')
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
        vbox2.setStretch(0,10)
        vbox2.setStretch(1,90)
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
        stitch_dir = QFileDialog.getExistingDirectory(self,'选择要拼接的文件夹','./')
        out_dir = QFileDialog.getExistingDirectory(self, '选择输出文件夹', './')
        self.label2.setText('开始拼接图像，请等待.....')
        self.label2.repaint()
        filenames = image_func.scan_directory(stitch_dir)
        print(len(filenames))
        images = image_func.load_images(filenames)
        image_func.image_stitch(images,out_dir)
        self.label2.setText('拼接成功!')
        self.label2.repaint()
        #display the image
        out_jpg = QPixmap(out_dir+'/stitched.jpg',)
        self.label4.setImage(out_jpg)
        self.label4.repaint()
        #scale_jpg = out_jpg.scaledToHeight(self.label4.height(),Qt.KeepAspectRatio)
        #scale_jpg = out_jpg.scaled(self.label4.size(),Qt.KeepAspectRatio)
        #self.label4.setPixmap(scale_jpg)
        #self.label4.setScaledContents(True)

    def setLabel4_backgroud(self):
        pe = QPalette()
        pe.setColor(QPalette.Background,Qt.black)
        self.label4.setAutoFillBackground(True)
        self.label4.setPalette(pe)




    def OnClickButton1(self):
        print("clicked")

    def OnClickButton2(self):
        print("clicked")


    def OnClickButton3(self):
        print("clicked")






app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

