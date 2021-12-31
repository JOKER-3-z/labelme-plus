from qtpy.QtCore import (
    Qt,
    QByteArray,
    QVariant,
    QCoreApplication,
    QThread,
    Signal
)
from qtpy.QtGui import QImage, QPixmap
from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QMainWindow, QMessageBox, QTableWidgetItem
from widget import AnnotationScene, AnnotationView

from controller import InteractiveController
import cv2
import numpy as np
import json

class ModelThread(QThread):
    _signal = Signal(dict)

    def __init__(self, controller, param_path):
        super().__init__()
        self.controller = controller
        self.param_path = param_path

    def run(self):
        success, res = self.controller.setModel(self.param_path, False)
        self._signal.emit(
            {"success": success, "res": res, "param_path": self.param_path}
        )



class App(QMainWindow):


    def __init__(self, parent=None):
        super(App, self).__init__(parent)


        CentralWidget = QtWidgets.QWidget(self)
        CentralWidget.setObjectName("CentralWidget")
        self.setCentralWidget(CentralWidget)
        ## -- 图形区域 --
        ImageRegion = QtWidgets.QHBoxLayout(CentralWidget)
        ImageRegion.setObjectName("ImageRegion")
        # 滑动区域
        self.scrollArea = QtWidgets.QScrollArea(CentralWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        ImageRegion.addWidget(self.scrollArea)
        # 图形显示
        self.scene = AnnotationScene()
        self.scene.addPixmap(QtGui.QPixmap())
        self.canvas = AnnotationView(self.scene, self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setAutoFillBackground(False)
        self.canvas.setStyleSheet("background-color: White")
        self.canvas.setObjectName("canvas")
        self.scrollArea.setWidget(self.canvas)


        self.predictor_params = {
            "brs_mode": "NoBRS",
            "with_flip": False,
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (400, 400),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {
                "net_clicks_limit": None,
                "max_size": 800,
                "with_mask": True,
            },
        }


        self.controller = InteractiveController(
            predictor_params=self.predictor_params,
            prob_thresh=0.5   ##self.segThresh,
        )
        # self.image = None

        param_path = "/mnt/hdd/EISeg/weights/static_hrnet18s_ocr48_human/static_hrnet18s_ocr48_human/static_hrnet18s_ocr48_human.pdiparams"
        self.load_thread = ModelThread(self.controller, param_path)
        # self.load_thread._signal.connect(self.__change_model_callback)
        self.load_thread.start()

        self.annImage = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.annImage)

        self.opacity = 0.5
        self.clickRadius = 3
        self.scene.clickRequest.connect(self.canvasClick)

        label = {"id": 0,
         "name": "person",
         "color": [255, 0, 0]}
        self.controller.setLabelList(json.dumps([label]))
        self.controller.setCurrLabelIdx(1)

        a = QtWidgets.QAction("finish", self)
        a.setShortcut("Space")
        a.triggered.connect(self.finishObject)
        a.setEnabled(True)
        self.scrollArea.addAction(a)

    def loadImage(self, path):
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            image = image[:, :, ::-1]  # BGR转RGB
            self.controller.setImage(image)
            # self.image = image
            self.updateImage()
        except Exception as e:
            print(e)


    # def showImage(self):
    #     if self.image is not None:
    #         print(self.image.shape)
    #         self.updateImage()
    #     else:
    #         print("no image")

    def updateImage(self, reset_canvas=False):
        if not self.controller:
            return
        image = self.controller.get_visualization(
            alpha_blend=self.opacity,
            click_radius=self.clickRadius,
        )
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        # if reset_canvas:
        #     self.resetZoom(width, height)
        self.annImage.setPixmap(QPixmap(image))

    def canvasClick(self, x, y, isLeft):
        c = self.controller
        if c.image is None:
            return
        if not c.inImage(x, y):
            return
        if not c.modelSet:
            self.warn(self.tr("未选择模型", self.tr("尚未选择模型，请先在右上角选择模型")))
            return

        # if self.status == self.IDILE:
        #     return
        currLabel = self.controller.curr_label_number
        # if not currLabel or currLabel == 0:
        # if not currLabel:
        #     self.warn(self.tr("未选择当前标签"), self.tr("请先在标签列表中单击点选标签"))
        #     return

        self.controller.addClick(x, y, isLeft)
        self.updateImage()


    def finishObject(self):
        print("to finish object")
        if not self.controller: #or self.image is None:
            return
        current_mask, curr_polygon = self.controller.finishObject(
            building=True)
        print(curr_polygon)
        self.updateImage()

        # if curr_polygon is not None:
        #     self.updateImage()
        #     if current_mask is not None:
        #         # current_mask = current_mask.astype(np.uint8) * 255
        #         # polygon = util.get_polygon(current_mask)
        #         color = self.controller.labelList[0].color
        #         self.createPoly(curr_polygon, color)
        # # 状态改变
        # if self.status == self.EDITING:
        #     self.status = self.ANNING
        #     for p in self.scene.polygon_items:
        #         p.setAnning(isAnning=True)
        # else:
        #     self.status = self.EDITING
        #     for p in self.scene.polygon_items:
        #         p.setAnning(isAnning=False)
        # self.getMask()


    # # 多边形标注
    # def createPoly(self, curr_polygon, color):
    #     if curr_polygon is None:
    #         return
    #     for points in curr_polygon:
    #         if len(points) < 3:
    #             continue
    #         poly = PolygonAnnotation(
    #             self.controller.labelList[self.currLabelIdx].idx,
    #             self.controller.image.shape,
    #             self.delPolygon,
    #             self.setDirty,
    #             color,
    #             color,
    #             self.opacity,
    #         )
    #         poly.labelIndex = self.controller.labelList[self.currLabelIdx].idx
    #         self.scene.addItem(poly)
    #         self.scene.polygon_items.append(poly)
    #         for p in points:
    #             poly.addPointLast(QtCore.QPointF(p[0], p[1]))
    #         self.setDirty(True)


    # 图片/标签 io
    # def getMask(self):
    #     if not self.controller or self.controller.image is None:
    #         return
    #     s = self.controller.imgShape
    #     pesudo = np.zeros([s[0], s[1]])
    #     # 覆盖顺序，从上往下
    #     # TODO: 是标签数值大的会覆盖小的吗?
    #     # A: 是列表中上面的覆盖下面的，由于标签可以移动，不一定是大小按顺序覆盖
    #     # RE: 我们做医学的时候覆盖比较多，感觉一般是数值大的标签覆盖数值小的标签。按照上面覆盖下面的话可能跟常见的情况正好是反过来的，感觉可能从下往上覆盖会比较好
    #     len_lab = self.labelListTable.rowCount()
    #     for i in range(len_lab - 1, -1, -1):
    #         idx = int(self.labelListTable.item(len_lab - i - 1, 0).text())
    #         for poly in self.scene.polygon_items:
    #             if poly.labelIndex == idx:
    #                 pts = np.int32([np.array(poly.scnenePoints)])
    #                 cv2.fillPoly(pesudo, pts=pts, color=idx)
    #     return pesudo



from qtpy.QtWidgets import QApplication
import sys


app = QApplication(sys.argv)


window = App()  # 创建对象

print("to load image")
window.loadImage("demo.jpg")
print("finiishi load image")
# window.showImage()



window.showMaximized()  # 全屏显示窗口
# 加载近期模型
QApplication.processEvents()

sys.exit(app.exec())
