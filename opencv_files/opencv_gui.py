from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
import utils


fil = utils.Filters()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1048, 721)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.EdgeDetectBox = QtWidgets.QGroupBox(self.centralwidget)
        self.EdgeDetectBox.setGeometry(QtCore.QRect(100, 100, 390, 500))
        self.EdgeDetectBox.setObjectName("EdgeDetectBox")
        self.gauss_blur_button = QtWidgets.QPushButton(self.EdgeDetectBox)
        self.gauss_blur_button.setGeometry(QtCore.QRect(50, 50, 300, 50))
        self.gauss_blur_button.setObjectName("gauss_blur_button")
        self.sobel_X_button = QtWidgets.QPushButton(self.EdgeDetectBox)
        self.sobel_X_button.setGeometry(QtCore.QRect(50, 150, 300, 50))
        self.sobel_X_button.setObjectName("sobel_X_button")
        self.sobel_Y_button = QtWidgets.QPushButton(self.EdgeDetectBox)
        self.sobel_Y_button.setGeometry(QtCore.QRect(50, 250, 300, 50))
        self.sobel_Y_button.setObjectName("sobel_Y_button")
        self.magnitude_button = QtWidgets.QPushButton(self.EdgeDetectBox)
        self.magnitude_button.setGeometry(QtCore.QRect(50, 350, 300, 50))
        self.magnitude_button.setObjectName("magnitude_button")
        self.TransformationBox = QtWidgets.QGroupBox(self.centralwidget)
        self.TransformationBox.setGeometry(QtCore.QRect(550, 100, 390, 500))
        self.TransformationBox.setObjectName("TransformationBox")
        self.resize_button = QtWidgets.QPushButton(self.TransformationBox)
        self.resize_button.setGeometry(QtCore.QRect(50, 50, 300, 50))
        self.resize_button.setIconSize(QtCore.QSize(30, 30))
        self.resize_button.setObjectName("resize_button")
        self.translate_button = QtWidgets.QPushButton(self.TransformationBox)
        self.translate_button.setGeometry(QtCore.QRect(50, 150, 300, 50))
        self.translate_button.setIconSize(QtCore.QSize(30, 30))
        self.translate_button.setObjectName("translate_button")
        self.rotate_button = QtWidgets.QPushButton(self.TransformationBox)
        self.rotate_button.setGeometry(QtCore.QRect(50, 250, 300, 50))
        self.rotate_button.setIconSize(QtCore.QSize(30, 30))
        self.rotate_button.setObjectName("rotate_button")
        self.shear_button = QtWidgets.QPushButton(self.TransformationBox)
        self.shear_button.setGeometry(QtCore.QRect(50, 350, 300, 50))
        self.shear_button.setIconSize(QtCore.QSize(30, 30))
        self.shear_button.setObjectName("shear_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.EdgeDetectBox.setTitle(_translate("MainWindow", "Edge Detection"))
        self.gauss_blur_button.setText(_translate("MainWindow", "Gaussian Blur"))
        self.sobel_X_button.setText(_translate("MainWindow", "Sobel X"))
        self.sobel_Y_button.setText(_translate("MainWindow", "Sobel Y"))
        self.magnitude_button.setText(_translate("MainWindow", "Magnitude"))
        self.TransformationBox.setTitle(_translate("MainWindow", "Transformation"))
        self.resize_button.setText(_translate("MainWindow", "Resize"))
        self.translate_button.setText(_translate("MainWindow", "Translate"))
        self.rotate_button.setText(_translate("MainWindow", "Rotate and Scale"))
        self.shear_button.setText(_translate("MainWindow", "Shear"))

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.gauss_blur_button.clicked.connect(self.something)

        self.sobel_X_button.clicked.connect(self.x_sobel)

        self.sobel_Y_button.clicked.connect(self.y_sobel)

    @QtCore.pyqtSlot()
    def browse_img(self):
        image = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile' ,'', "Image file(*.png *.jpg *.bmp)")
        path = image[0]
        return path

    def something(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        fil.blur_img(image)

    def x_sobel(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        fil.get_x_edges(image)

    def y_sobel(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        fil.get_y_edges(image)

    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
