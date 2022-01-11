from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import QT5
from labelme.shape import Shape
import labelme.utils

from labelme.pose_config import *
from labelme.baidu import get_annoatation_from_baidu, visible_threshold

from eiseg.controller import InteractiveController


import os
import json
import cv2
import numpy as np
from labelme.utils import calIOU

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0

class ModelThread(QtCore.QThread):
    _signal = QtCore.Signal(dict)

    def __init__(self, controller, param_path):
        super().__init__()
        self.controller = controller
        self.param_path = param_path

    def run(self):
        success, res = self.controller.setModel(self.param_path, False)
        self._signal.emit(
            {"success": success, "res": res, "param_path": self.param_path}
        )


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    newPoseShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    segRequest = QtCore.Signal(int, int, bool)



    CREATE, EDIT = 0, 1
    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

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
            prob_thresh=0.5  ##self.segThresh,
        )
        here = os.path.dirname(os.path.abspath(__file__))
        param_path = "../weights/static_hrnet18s_ocr48_human.pdiparams"
        param_path = os.path.join(here, param_path)
        param_path = os.path.abspath(param_path)
        self.load_thread = ModelThread(self.controller, param_path)
        self.load_thread.start()
        self.opacity = 0.5
        self.clickRadius = 3

        self.segRequest.connect(self.canvasClick)
        label = {"id": 0,
                 "name": "person",
                 "color": [255, 0, 0]}
        self.controller.setLabelList(json.dumps([label]))
        self.controller.setCurrLabelIdx(1)
        self.controller.filterLargestCC(True)




    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "pose",
            "rectangle_selection",
            "pose_by_baidu",
            "seg_by_eiseg"

        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None


    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()

        # Polygon drawing.
        if self.drawing() or self.createMode == "rectangle_selection":
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode in ["rectangle","rectangle_selection", "pose_by_baidu"]:
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "pose":
                self.line.points = [self.current[0], pos]
                self.line.close()
                self.current.pose = {}
                self.current.label = "pose"
                start = self.line.points[0]
                end = self.line.points[1]
                width = end.x() - start.x()
                height = end.y() - start.y()
                for key, tpoint in pose_define["location"].items():
                    self.current.pose[key] = QtCore.QPoint(int(tpoint[0]* width + start.x()), int(tpoint[1]*height + start.y()))

            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.repaint()
            self.current.highlightClear()
            return
        # else:
        #     self.line.shape_type = self.createMode
        #
        #     self.overrideCursor(CURSOR_DRAW)
        #     if not self.current:
        #         return
        #     if self.createMode == "rectangle_selection":
        #         self.line.points = [self.current[0], pos]
        #         self.line.close()
        #     return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [
                    s.copy() for s in self.selectedShapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        point_shape = [s for s in self.shapes if (self.isVisible(s) and s.shape_type != "point")]
        point_shape.extend([s for s in self.shapes if (self.isVisible(s) and s.shape_type == "point")])

        # for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
        # self.hShape.highlightClear()
        # self.selectedShapes.clear()
        for shape in reversed(point_shape):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale) if shape.shape_type != "point" else None
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                str_conf = ""
                if shape.confidence is not None:
                    str_conf = "conf:{:.3f}".format(shape.confidence)
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s' %s") %(shape.label, str_conf)
                )
                self.setStatusTip(self.toolTip())
                # shape.selected = True
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        if self.createMode == "seg_by_eiseg" and ev.button() in [QtCore.Qt.LeftButton, QtCore.Qt.RightButton]:
            isLeft = True if  ev.button() == QtCore.Qt.LeftButton else False
            self.segRequest.emit(pos.x(), pos.y(), isLeft)
            return

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing() or self.createMode == "rectangle_selection":
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode in ["pose"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode in ["pose_by_baidu"]:
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode == "rectangle_selection":
                        self.current.points = self.line.points
                        point_shape = [s for s in self.shapes if (self.isVisible(s) and s.shape_type == "point")]
                        for shape in reversed(point_shape):
                            if self.isVisible(shape) and self.current.containsPoint(shape.points[0]):
                                self.setHiding()
                                if shape not in self.selectedShapes:
                                    self.selectionChanged.emit(self.selectedShapes + [shape])
                                    self.hShapeIsSelected = False
                                else:
                                    self.hShapeIsSelected = True
                                # self.calculateOffsets(point)
                        # self.deSelectShape()
                        self.current = None



                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                    self.selectedVertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None
                and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

        elif ev.button() == QtCore.Qt.MiddleButton:
            if not self.outOfPixmap(pos):
                # Create new shape.
                self.current = Shape(shape_type="rectangle_selection")
                self.current.addPoint(pos)
                self.createMode = "rectangle_selection"

                self.line.points = [pos, pos]
                self.setHiding()
                self.drawingPolygon.emit(True)
                self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton and self.createMode != "seg_by_eiseg":
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selectedShapesCopy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )


        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                self.shapesBackups[-1][index].points
                != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.canCloseShape()
            and len(self.current) > 3
        ):
            self.current.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            point_shape = [s for s in self.shapes if (self.isVisible(s) and s.shape_type != "point")]
            point_shape.extend([s for s in self.shapes if (self.isVisible(s) and s.shape_type == "point")])

            # for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            for shape in reversed(point_shape):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                            if shape.shape_type == "pose":
                                pose_shape = [shape for shape in shape.pose_shape]
                                self.selectionChanged.emit(
                                    self.selectedShapes + pose_shape
                                )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True


                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPoint(x1, y1), QtCore.QPoint(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                if shape in self.shapes:
                    self.shapes.remove(shape)
                    deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPoint(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale

        for shape in self.shapes:
            # if (shape.selected or not self._hideBackround) and self.isVisible(
            #     shape
            # ):
            if (not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def getMaxGroupId(self):
        maxGroupId = 0
        for item in self.shapes:
            if item.group_id is not None:
                maxGroupId = max(item.group_id, maxGroupId)
        return maxGroupId

    def updatePose(self):
        pose_shapes = [x for x in self.shapes if x.shape_type=="pose"]
        points = [x for x in self.shapes if x.shape_type == "point" and x.label in pose_define["keypoints"]]
        for pose_shape in pose_shapes:
            for point in points:
                if pose_shape.group_id == point.group_id:
                    pose_shape.pose_shape.append(point)


    def finalise(self):
        if self.createMode == "seg_by_eiseg":
            print("to finish object")
            if not self.controller:  # or self.image is None:
                return
            current_mask, curr_polygon = self.controller.finishObject(
                building=False)

            print(curr_polygon)
            if curr_polygon is None:
                return
            if len(curr_polygon) > 0:
                group_id = None
                bbox = cv2.boundingRect(np.array(curr_polygon[0]))
                # bbox = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]
                max_iou = 0
                for shape in [pose for pose in self.shapes if pose.label == "pose"]:
                    bbox_shape = [shape.points[0].x(),shape.points[0].y(),
                                  shape.points[1].x() - shape.points[0].x(),
                                  shape.points[1].y() - shape.points[0].y()
                                  ]
                    iou = calIOU(bbox, bbox_shape)
                    if  iou > 0.05 and iou > max_iou:
                        group_id = shape.group_id
                        max_iou = iou
                        # break




                current = Shape(shape_type="polygon", group_id=group_id, label="person")
                for point in curr_polygon[0]:
                    current.points.append(QtCore.QPoint(int(point[0]), int(point[1])))

            self.shapes.append(current)
            self.storeShapes()
            self.setHiding(False)

            self.newPoseShape.emit()
            self.update()

            # self.updateImage()
            return

        assert self.current
        self.current.close()
        if self.createMode == "pose_by_baidu":
            pixmap = self.pixmap.copy(self.current.points[0].x(), self.current.points[0].y(),
                                        self.current.points[1].x() - self.current.points[0].x(),
                                        self.current.points[1].y() - self.current.points[0].y(),
                                      )

            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.WriteOnly)
            pixmap.save(buffer, "jpg")
            encoded = buffer.data()   #.toBase64()
            # with open("tmp.jpg", 'wb') as fp:
            #     fp.write(encoded)
            baidu_anno = get_annoatation_from_baidu(encoded)
            print(baidu_anno)

            start = self.line.points[0]
            end = self.line.points[1]
            width = end.x() - start.x()
            height = end.y() - start.y()
            if "person_info" not in baidu_anno.keys():
                return
            for idx, person in enumerate(baidu_anno["person_info"]):
                # self.current.pose[key] = QtCore.QPoint(int(tpoint[0] * width + start.x()),
                #                                        int(tpoint[1] * height + start.y()))
                group_id = self.getMaxGroupId() + 1
                current = Shape(shape_type="pose", group_id=group_id)
                current.pose = {}
                current.label = "pose"
                bbox = person["location"]
                top_left = QtCore.QPoint(bbox["left"] + start.x(),bbox["top"] + start.y())
                bot_right = QtCore.QPoint(bbox["left"] + start.x() + bbox["width"],bbox["top"] + start.y() + bbox["height"])
                current.points = [top_left, bot_right]

                for part_name, info in person["body_parts"].items():
                    if part_name in pose_define["keypoints_priority"] or \
                            (part_name in pose_define["keypoints"] and info["score"] > visible_threshold):
                        point = Shape(shape_type="point", group_id=group_id, label=part_name, confidence=info["score"])
                        location = QtCore.QPoint(info["x"] + start.x(), info["y"] + start.y())
                        point.current = [location]
                        point.points = [location]
                        self.shapes.append(point)
                        current.pose_shape.append(point)
                        self.newPoseShape.emit()

                self.shapes.append(current)
                self.storeShapes()
                self.setHiding(False)

                self.newPoseShape.emit()
                self.update()
                self.current = None
            return




        if self.createMode == "pose":
            group_id = self.getMaxGroupId() + 1
            self.current.group_id = group_id
            for key, location in self.current.pose.items():
                point = Shape(shape_type="point", group_id=group_id, label=key)
                point.current = [location]
                point.points = [location]
                self.shapes.append(point)
                self.current.pose_shape.append(point)
                self.newPoseShape.emit()
            self.current.pose = {}







        # else:
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None

        self.setHiding(False)
        if self.createMode != "pose":
            self.newShape.emit()
        else:
            self.newPoseShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPoint(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPoint(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(
                self.selectedShapes, self.prevPoint + offset
            )
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()

        if self.createMode == "seg_by_eiseg" and key == QtCore.Qt.Key_Space:
            self.finalise()
            return



        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPoint(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPoint(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPoint(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPoint(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def getLastLabel(self):
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle", "pose"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

    def canvasClick(self, x, y, isLeft):
        c = self.controller
        if c.image is None:
            return
        if not c.inImage(x, y):
            return
        if not c.modelSet:
            self.warn(self.tr("未选择模型", self.tr("尚未选择模型，请先在右上角选择模型")))
            return

        res = self.controller.addClick(x, y, isLeft)
        # print(res)
        self.updateImage()

    def updateImage(self, reset_canvas=False):
        if not self.controller:
            return
        image = self.controller.get_visualization(
            alpha_blend=self.opacity,
            click_radius=self.clickRadius,
        )
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # if reset_canvas:
        #     self.resetZoom(width, height)
        self.loadPixmap(QtGui.QPixmap(image), False)