from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
import utils
import cv2 as cv


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

        self.gauss_blur_button.clicked.connect(self.show_blur)

        self.sobel_X_button.clicked.connect(self.filter_x_edges)

        self.sobel_Y_button.clicked.connect(self.filter_y_edges)

        self.magnitude_button.clicked.connect(self.show_magnitude)

        self.resize_button.clicked.connect(self.resize_image) 

        self.translate_button.clicked.connect(self.translate_image)

        self.rotate_button.clicked.connect(self.rotate_and_scale)

        self.shear_button.clicked.connect(self.shear)

    @QtCore.pyqtSlot()
    def browse_img(self):
        image = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile' ,'', "Image file(*.png *.jpg *.bmp)")
        path = image[0]
        return path

    def show_blur(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        b = fil.apply_gauss(image)
        cv.imshow("Gauss Blur", b)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def filter_x_edges(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        x = fil.apply_x_sobel(image)
        cv.imshow("X Sobel Filter", x)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def filter_y_edges(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        y = fil.apply_y_sobel(image)
        cv.imshow("Y Sobel Filter", y)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_magnitude(self):
        path = self.browse_img()
        image = utils.load_gray_image(path)
        x = fil.apply_x_sobel(image)
        y = fil.apply_y_sobel(image)
        m = utils.get_magnitude(x, y)
        cv.imshow("Magnitude", m)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def resize_image(self):
        path = self.browse_img()
        resized_img = utils.apply_resize(path)
        cv.imshow("Resize and Translate", resized_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def translate_image(self):
        path = self.browse_img()
        translated_img = utils.apply_translation(path)
        cv.imshow("Translate and Duplicate", translated_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def rotate_and_scale(self):
        path = self.browse_img()
        rotated_img = utils.apply_rotation_and_scaling(path)
        cv.imshow("Rotation and Scaling", rotated_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def shear(self):
        path = self.browse_img()
        shear_img = utils.apply_shear(path)
        cv.imshow("Shear", shear_img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
