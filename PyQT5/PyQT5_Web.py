import sys
from PyQt5.Qt import *
from PyQt5.QtWebEngineWidgets import * 
# If you have error this line
# try : pip install PyQtWebEngine
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

web = QWebEngineView()

web.load(QUrl("https://www.naver.com"))

web.show()

sys.exit(app.exec_())