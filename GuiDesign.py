from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setWindowTitle("图像处理程序 版本： 0.8")
        self.resize(640,480)
        self.initUI()

    def initUI(self):
        wlayout = QGridLayout()

        hsplitter = QSplitter(Qt.Horizontal)
        vsplitter = QSplitter(Qt.Vertical)


        hwg1 = QWidget()
        hwg2 = QWidget()
        vwg  = QWidget()
        hlayout1 = QVBoxLayout()
        hlayout2 = QVBoxLayout()
        vlayout = QHBoxLayout()

        hsplitter.addWidget(hwg1)
        hsplitter.addWidget(hwg2)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(vwg)


        hlayout1.addWidget(QPushButton('图像拼接'))
        hlayout1.addWidget(QPushButton('计算重心'))
        hlayout1.addWidget(QPushButton('颜色统计图'))
        hlayout1.addWidget(QPushButton('网格化'))

        hlayout2.addWidget(QPushButton('1'))

        hwg1.setLayout(hlayout1)
        hwg2.setLayout(hlayout2)
        vwg.setLayout(vlayout)



        wlayout.addWidget(vsplitter,0,0)
        # wlayout.addWidget(hwg2,0,1)
        # wlayout.addWidget(vwg,1,0)
        self.setLayout(wlayout)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

