#!/usr/bin/env python3
"""Entry point for the Langton's Ant simulator (Python port)."""
import sys

from PySide6.QtWidgets import QApplication

from gui.mainwindow import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
