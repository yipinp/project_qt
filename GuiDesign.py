from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 576

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


        vbox1 = QGridLayout()
        vbox1.setAlignment(Qt.AlignTop)
        label1 = QLabel('信息显示区')
        label1.setAlignment(Qt.AlignCenter)
        vbox1.addWidget(label1,0,0)
        hwg2.setLayout(vbox1)

        hsplitter.addWidget(hwg1)
        hsplitter.addWidget(hwg2)
        hsplitter.setStretchFactor(0,50)
        hsplitter.setStretchFactor(1,50)
        hsplitter.setAutoFillBackground(True)

        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(hsplitter)
        vbox2 = QGridLayout()
        hwg3.setLayout(vbox2)
        vbox2.setAlignment(Qt.AlignTop)
        label2 = QLabel('图片显示区')
        label2.setAlignment(Qt.AlignCenter)
        vbox2.addWidget(label2)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(hwg3)
        vsplitter.setStretchFactor(1,4)
        vsplitter.setStretchFactor(2,6)
        vsplitter.setAutoFillBackground(True)


        hbox.addWidget(vsplitter)
        wwg.setLayout(hbox)
        self.setCentralWidget(wwg)





app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

