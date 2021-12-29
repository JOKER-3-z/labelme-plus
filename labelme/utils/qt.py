from math import sqrt
import os.path as osp

import numpy as np

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


here = osp.dirname(osp.abspath(__file__))


def newIcon(icon):
    icons_dir = osp.join(here, "../icons")
    return QtGui.QIcon(osp.join(":/", icons_dir, "%s.png" % icon))


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(" ", "\n"))
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r"^[^ \t].+"), None)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return 0
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fmtShortcut(text):
    mod, key = text.split("+", 1)
    return "<b>%s</b>+<b>%s</b>" % (mod, key)

def QImageToArray(image):
   # # Get the size of the current pixmap
   #  size = pixmap.size()
   #  h = size.width()
   #  w = size.height()
   #  print(h,w)
   #
   #  # # Get the QImage Item and convert it to a byte string
   #  qimg = pixmap.toImage()
   #  byte_str = qimg.bits().tobytes()
   #
   #  # # Using the np.frombuffer function to convert the byte string into an np array
   #  img = np.frombuffer(byte_str, dtype = np.uint8).reshape((w, h, 4))


    channels_count = int(image.depth()/8)
    size = image.size()
    b = image.bits()
    h = size.width()
    w = size.height()
    # t = QtGui.QImage()
    # t.depth()
   # sip.voidptr must know size to support python buffer interface
    b.setsize(h * w * channels_count)
    arr = np.frombuffer(b, np.uint8).reshape((w, h, channels_count))

    print(arr.shape, size,channels_count)
    return arr[:,:,:3]