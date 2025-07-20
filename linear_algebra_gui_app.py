import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSpinBox, QHBoxLayout, QScrollArea
)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.linalg


class LinearAlgebraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔥 Linear Algebra App — SVD, Inverse, Plotting")
        self.setGeometry(100, 100, 850, 700)
        self.matrix_inputs = []
        self.setup_ui()
        self.animate_window()

    def setup_ui(self):
        # Dark theme
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)

        self.layout = QVBoxLayout(self)

        # Title
        title = QLabel("🔥 Linear Algebra Web App — SVD, Inverse, Plotting")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: white; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        # Operation dropdown
        self.label = QLabel("Select Operation")
        self.label.setFont(QFont("Segoe UI", 12))
        self.label.setStyleSheet("color: white;")
        self.layout.addWidget(self.label)

        self.operation_box = QComboBox()
        self.operation_box.addItems([
            "Inverse", "Determinant", "Rank",
            "Eigenvalues", "SVD", "LU Decomposition", "3D Plot"
        ])
        self.operation_box.setStyleSheet("padding: 8px; border-radius: 5px;")
        self.layout.addWidget(self.operation_box)

        # Matrix count spinner
        self.matrix_count_layout = QHBoxLayout()
        self.matrix_count_label = QLabel("How many matrices?")
        self.matrix_count_label.setFont(QFont("Segoe UI", 12))
        self.matrix_count_label.setStyleSheet("color: white;")
        self.matrix_count_spinner = QSpinBox()
        self.matrix_count_spinner.setMinimum(1)
        self.matrix_count_spinner.setMaximum(10)
        self.matrix_count_spinner.setValue(1)
        self.matrix_count_spinner.valueChanged.connect(self.update_matrix_tables)
        self.matrix_count_spinner.setStyleSheet("padding: 6px; border-radius: 4px;")
        self.matrix_count_layout.addWidget(self.matrix_count_label)
        self.matrix_count_layout.addWidget(self.matrix_count_spinner)
        self.layout.addLayout(self.matrix_count_layout)

        # Scrollable matrix input area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")

        self.scroll_widget = QWidget()
        self.matrix_inputs_layout = QVBoxLayout(self.scroll_widget)
        self.matrix_inputs_layout.setContentsMargins(0, 0, 0, 0)
        self.matrix_inputs_layout.setSpacing(12)

        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)

        # Compute button
        self.compute_button = QPushButton("Compute")
        self.compute_button.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 10px; font-size: 14px; border-radius: 5px;"
        )
        self.compute_button.clicked.connect(self.compute_result)
        self.layout.addWidget(self.compute_button)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setFont(QFont("Courier", 11))
        self.result_label.setStyleSheet("color: lightgreen; margin-top: 10px;")
        self.layout.addWidget(self.result_label)

        self.update_matrix_tables()

    def animate_window(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(800)
        self.animation.setStartValue(QRect(0, 0, 100, 30))
        self.animation.setEndValue(QRect(100, 100, 850, 700))
        self.animation.start()

    def update_matrix_tables(self):
        for widget in self.matrix_inputs:
            self.matrix_inputs_layout.removeWidget(widget)
            widget.setParent(None)
        self.matrix_inputs.clear()

        for idx in range(self.matrix_count_spinner.value()):

            table = QTableWidget(3, 3)
            table.setFont(QFont("Consolas", 12))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.matrix_inputs_layout.addWidget(table)
            self.matrix_inputs.append(table)

    def compute_result(self):
        try:
            matrices = []
            for table in self.matrix_inputs:
                matrix = np.array([[float(table.item(i, j).text()) if table.item(i, j) else 0
                                    for j in range(table.columnCount())]
                                   for i in range(table.rowCount())])
                matrices.append(matrix)

            operation = self.operation_box.currentText()

            if operation == "Inverse":
                result = np.linalg.inv(matrices[0])
            elif operation == "Determinant":
                result = np.linalg.det(matrices[0])
            elif operation == "Rank":
                result = np.linalg.matrix_rank(matrices[0])
            elif operation == "Eigenvalues":
                result = np.linalg.eigvals(matrices[0])
            elif operation == "SVD":
                U, S, VT = np.linalg.svd(matrices[0])
                result = f"U:\n{U}\n\nS:\n{S}\n\nVT:\n{VT}"
            elif operation == "LU Decomposition":
                P, L, U = scipy.linalg.lu(matrices[0])
                result = f"P:\n{P}\n\nL:\n{L}\n\nU:\n{U}"
            elif operation == "3D Plot":
                self.plot_transformed_vectors(matrices)
                return
            else:
                result = "Unsupported Operation"

            self.result_label.setText(str(result))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_transformed_vectors(self, matrices):
        if len(matrices) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Please input at least 2 matrices:\n1. Transform matrix\n2. Vectors to be transformed.")
            return

        transform = matrices[0]
        vectors = matrices[1:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        origin = [0, 0, 0]

        for vec_matrix in vectors:
            flat_vecs = vec_matrix.reshape(-1, 3)
            for vec in flat_vecs:
                transformed_vec = transform @ vec
                ax.quiver(*origin, *transformed_vec, arrow_length_ratio=0.1, color='r')

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_title("Transformed 3D Vectors")
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinearAlgebraApp()
    window.show()
    sys.exit(app.exec_())