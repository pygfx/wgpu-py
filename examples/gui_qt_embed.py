"""
An example demonstrating a qt app with a wgpu viz inside.
If needed, change the PySide6 import to e.g. PyQt6, PyQt5, or PySide2.

"""

# ruff: noqa: N802
# run_example = false

import time
import importlib

from cube import setup_drawing_sync

# For the sake of making this example Just Work, we try multiple QT libs
for lib in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
    try:
        QtWidgets = importlib.import_module(".QtWidgets", lib)
        break
    except ModuleNotFoundError:
        pass

from rendercanvas.qt import QRenderWidget  # noqa: E402


class ExampleWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(640, 480)
        self.setWindowTitle("wgpu cube embedded in a qt app")

        splitter = QtWidgets.QSplitter()

        self.button = QtWidgets.QPushButton("Hello world", self)
        self.button2 = QtWidgets.QPushButton("pause", self)
        self.canvas = QRenderWidget(splitter, update_mode="continuous")
        self.output = QtWidgets.QTextEdit(splitter)

        self.button.clicked.connect(self.whenButtonClicked)
        self.button2.clicked.connect(self.whenButton2Clicked)

        splitter.addWidget(self.canvas)
        splitter.addWidget(self.output)
        splitter.setSizes([400, 300])

        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.button2)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        self.show()
        self._paused = False

    def addLine(self, line):
        t = self.output.toPlainText()
        t += "\n" + line
        self.output.setPlainText(t)

    def whenButtonClicked(self):
        self.addLine(f"Clicked at {time.time():0.1f}")

    def whenButton2Clicked(self):
        # showcases how rendercanvas allows changes to sheduling interactively
        if self._paused:
            self.canvas.set_update_mode("continuous", max_fps=60)
            self.button2.setText("pause")
            self._paused = False
        else:
            # note: the cube example bases rotation on unix time, which we don't pause with this button
            # with "ondemand", resize events such as the window or the splitter will still trigger a draw!
            self.canvas.set_update_mode("ondemand")
            self.button2.setText("resume")
            self._paused = True


app = QtWidgets.QApplication([])
example = ExampleWidget()

draw_frame = setup_drawing_sync(example.canvas)
example.canvas.request_draw(draw_frame)

# Enter Qt event loop (compatible with qt5/qt6)
app.exec() if hasattr(app, "exec") else app.exec_()
