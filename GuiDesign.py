from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.initUI()

    def initUI(self):
        exitAct = QAction(QIcon('exit.png'),'&Exit',self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit Application')


        self.statusBar().showMessage("Ready")
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(exitAct)


        self.setWindowTitle("version 0.8")
        self.setGeometry(300,300,300,300)
        self.setGeometry(300,300,300,300)
        self.show()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()

