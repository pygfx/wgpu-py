"""
An example demonstrating a qt app with a wgpu viz inside.

This is the same as the ``gui_qt_embed.py`` example, except this uses
the asyncio compatible mode that was introduced in Pyside 6.6.
"""

# ruff: noqa: N802
# run_example = false

import time
import asyncio

from PySide6 import QtWidgets, QtAsyncio
from wgpu.gui.qt import WgpuWidget
from triangle import setup_drawing_sync


class ExampleWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle("wgpu triangle embedded in a qt app")

        splitter = QtWidgets.QSplitter()

        self.button = QtWidgets.QPushButton("Hello world", self)
        self.canvas = WgpuWidget(splitter)
        self.output = QtWidgets.QTextEdit(splitter)

        # self.button.clicked.connect(self.whenButtonClicked)
        self.button.clicked.connect(
            lambda: asyncio.ensure_future(self.whenButtonClicked())
        )

        splitter.addWidget(
            self.canvas,
        )
        splitter.addWidget(self.output)
        splitter.setSizes([400, 300])

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.button, 0)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        self.show()

    def addLine(self, line):
        t = self.output.toPlainText()
        t += "\n" + line
        self.output.setPlainText(t)

    async def whenButtonClicked(self):
        self.addLine("Waiting 1 sec ...")
        await asyncio.sleep(1)
        self.addLine(f"Clicked at {time.time():0.1f}")


app = QtWidgets.QApplication([])
example = ExampleWidget()

draw_frame = setup_drawing_sync(example.canvas)
example.canvas.request_draw(draw_frame)

# Enter Qt event loop the asyncio-compatible way
QtAsyncio.run()
