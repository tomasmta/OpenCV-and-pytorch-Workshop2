from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
import vgg19utils


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setIconSize(QtCore.QSize(100, 100))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(120, 90, 280, 371))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.train_imgs_button = QtWidgets.QPushButton(self.groupBox)
        self.train_imgs_button.setGeometry(QtCore.QRect(50, 40, 200, 50))
        self.train_imgs_button.setObjectName("train_ims_button")
        self.structure_model_button = QtWidgets.QPushButton(self.groupBox)
        self.structure_model_button.setGeometry(QtCore.QRect(50, 100, 200, 50))
        self.structure_model_button.setObjectName("structure_model")
        self.augmentation_button = QtWidgets.QPushButton(self.groupBox)
        self.augmentation_button.setGeometry(QtCore.QRect(50, 160, 200, 50))
        self.augmentation_button.setObjectName("augmentation_button")
        self.metrics_button = QtWidgets.QPushButton(self.groupBox)
        self.metrics_button.setGeometry(QtCore.QRect(50, 220, 200, 50))
        self.metrics_button.setObjectName("metrics_button")
        self.inference_button = QtWidgets.QPushButton(self.groupBox)
        self.inference_button.setGeometry(QtCore.QRect(50, 280, 200, 50))
        self.inference_button.setObjectName("inference_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.train_imgs_button.setText(_translate("MainWindow", "Show Train Images"))
        self.structure_model_button.setText(_translate("MainWindow", "Show Model Structure"))
        self.augmentation_button.setText(_translate("MainWindow", "Data Augmentation"))
        self.metrics_button.setText(_translate("MainWindow", "Accuracy and Loss"))
        self.inference_button.setText(_translate("MainWindow", "Inference"))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.train_imgs_button.clicked.connect(vgg19utils.show_images)

        self.structure_model_button.clicked.connect(vgg19utils.show_model_structure)

        self.metrics_button.clicked.connect(vgg19utils.show_metrics)

        self.augmentation_button.clicked.connect(self.transform_img)



    @QtCore.pyqtSlot()
    def browse_img(self):
        image = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile' ,'', "Image file(*.png *.jpg *.bmp)")
        path = image[0]
        return path

    def transform_img(self):
        path = self.browse_img()
        vgg19utils.plot_transforms(path)
    



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
