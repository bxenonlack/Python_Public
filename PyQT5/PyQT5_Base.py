import sys
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication

class Main(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.Init_UI()

    def Init_UI(self):
        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('Title')

        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Main = Main()
    sys.exit(app.exec_())