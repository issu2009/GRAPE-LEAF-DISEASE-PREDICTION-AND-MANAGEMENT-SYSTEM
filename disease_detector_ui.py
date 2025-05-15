from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QWidget, QFileDialog, QMessageBox, 
                            QFrame, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize
import sys
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path

class StyleSheet:
    MAIN_STYLE = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            min-width: 150px;
        }
        QPushButton:hover {
            background-color: #1976D2;
        }
        QPushButton:disabled {
            background-color: #BDBDBD;
        }
        QLabel {
            font-size: 14px;
            color: #333333;
        }
        QProgressBar {
            border: 2px solid #2196F3;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2196F3;
        }
    """

class GrapeDiseaseDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        self.disease_precautions = {
            'ESCA': """
            Precautions for ESCA (Grapevine Trunk Disease):
            1. Remove and destroy infected vines
            2. Protect pruning wounds with wound sealant
            3. Sanitize pruning tools between cuts
            4. Avoid pruning during wet weather
            5. Consider preventive fungicide treatments
            """,
            
            'Leaf Blight': """
            Precautions for Leaf Blight:
            1. Maintain good air circulation between vines
            2. Apply copper-based fungicides
            3. Remove affected leaves promptly
            4. Avoid wetting leaves during irrigation
            5. Practice crop rotation when possible
            """,
            
            'Healthy': """
            Maintenance Tips for Healthy Vines:
            1. Regular pruning and training
            2. Proper irrigation management
            3. Balanced fertilization
            4. Regular monitoring for early detection
            5. Maintain good vineyard hygiene
            """
        }

    def initUI(self):
        self.setWindowTitle("Grape Leaf Disease Detector")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet(StyleSheet.MAIN_STYLE)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Create title label
        title_label = QLabel("Grape Leaf Disease Detector")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            color: #1565C0;
            font-weight: bold;
            margin: 20px;
        """)

        # Create image frame
        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #E0E0E0;
                border-radius: 10px;
            }
        """)
        image_layout = QVBoxLayout(self.image_frame)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        image_layout.addWidget(self.image_label)

        # Create status label
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 16px;
            color: #666666;
            padding: 10px;
        """)

        # Create select button
        self.select_button = QPushButton("Select Image")
        self.select_button.setIcon(self.style().standardIcon(self.style().SP_DialogOpenButton))
        self.select_button.clicked.connect(self.select_image)
        self.select_button.setEnabled(False)

        # Create result frame
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #E0E0E0;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        result_layout = QVBoxLayout(result_frame)
        
        # Create result label
        self.result_label = QLabel("Select an image for disease detection")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 16px;
            color: #333333;
            padding: 10px;
        """)
        result_layout.addWidget(self.result_label)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Create precautions text area
        self.precautions_label = QLabel("Disease Management:")
        self.precautions_label.setStyleSheet("""
            font-size: 16px;
            color: #1565C0;
            font-weight: bold;
        """)
        
        self.precautions_text = QLabel()
        self.precautions_text.setWordWrap(True)
        self.precautions_text.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                color: #333333;
                line-height: 1.4;
            }
        """)

        # Add widgets to main layout
        layout.addWidget(title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.image_frame)
        layout.addWidget(self.progress_bar)
        layout.addWidget(result_frame)
        layout.addWidget(self.precautions_label)
        layout.addWidget(self.precautions_text)

    def loadModel(self):
        try:
            # Try to find the model in possible locations
            possible_paths = [
                Path('runs/grape_disease_model/weights/best.pt'),  # Main path first
                Path('runs/detect/grape_disease_model/weights/best.pt'),
                Path(r'D:\Nuthan\grape_leap_detector\runs\grape_disease_model\weights\best.pt'),
                Path(r'D:\Nuthan\grape_leap_detector\runs\detect\grape_disease_model\weights\best.pt')
            ]

            # Find first existing model path
            model_path = next((p for p in possible_paths if p.exists()), None)

            if model_path:
                print(f"Loading model from: {model_path}")
                self.model = YOLO(str(model_path))
                
                # Update model class names through dictionary update instead of direct assignment
                self.class_names = {
                    0: 'ESCA',
                    1: 'Healthy',
                    2: 'Leaf Blight'
                }
                
                # Verify model loaded successfully
                self.status_label.setText(f"✅ Model loaded successfully")
                self.select_button.setEnabled(True)
                print("Model loaded successfully")
                
            else:
                raise FileNotFoundError("Could not find trained model in any location")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.status_label.setText("❌ Error loading model!")
            self.select_button.setEnabled(False)

    def select_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Image", 
                "", 
                "Image Files (*.png *.jpg *.jpeg)"
            )
            if file_name:
                # Show progress bar
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                QApplication.processEvents()

                # Load and display image
                image = cv2.imread(file_name)
                if image is None:
                    raise ValueError("Failed to load image")
                
                self.progress_bar.setValue(30)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Update status
                self.status_label.setText("🔍 Processing image...")
                self.progress_bar.setValue(50)
                QApplication.processEvents()
                
                # Different confidence thresholds for each class
                class_conf_thresholds = {
                    0: 0.3,  # ESCA
                    1: 0.3,  # Healthy
                    2: 0.2  # Lower threshold for Leaf Blight
                }
                
                # Predict disease
                results = self.model(image)
                
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    if len(boxes) > 0:
                        confidences = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy()
                        
                        # Filter predictions based on class-specific thresholds
                        valid_detections = []
                        for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
                            if conf >= class_conf_thresholds[int(cls_id)]:
                                valid_detections.append(i)
                        
                        if valid_detections:
                            best_idx = valid_detections[np.argmax(confidences[valid_detections])]
                            # Get the detection with highest confidence
                            class_name = self.class_names[int(class_ids[best_idx])]  # Use our class names dictionary
                            conf = confidences[best_idx]
                            
                            # Format result text with emoji
                            disease_emoji = "🍃" if class_name == "Healthy" else "🔬"
                            result_text = f"{disease_emoji} Detected: {class_name}\n"
                            result_text += f"Confidence: {conf:.2%}\n"
                            
                            # Add debugging info
                            result_text += f"\nAll detections:\n"
                            for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
                                cls_name = self.model.names[int(cls_id)]
                                result_text += f"{cls_name}: {conf:.2%}\n"
                            
                            self.result_label.setText(result_text)
                            
                            # Show precautions for detected disease
                            if class_name in self.disease_precautions:
                                self.precautions_text.setText(self.disease_precautions[class_name])
                                self.precautions_text.setStyleSheet("""
                                    QLabel {
                                        background-color: #f5f5f5;
                                        border: 1px solid #e0e0e0;
                                        border-radius: 5px;
                                        padding: 10px;
                                        font-size: 14px;
                                        color: #333333;
                                        line-height: 1.4;
                                    }
                                """)
                                if class_name == "Healthy":
                                    self.precautions_label.setStyleSheet("""
                                        font-size: 16px;
                                        color: #4CAF50;
                                        font-weight: bold;
                                    """)
                                else:
                                    self.precautions_label.setStyleSheet("""
                                        font-size: 16px;
                                        color: #f44336;
                                        font-weight: bold;
                                    """)
                        else:
                            self.result_label.setText("❓ No disease detected (No boxes)")
                    else:
                        self.result_label.setText("❓ No disease detected (No results)")

                # Display image with boxes
                if len(results) > 0:
                    # Draw boxes on image
                    result_plotted = results[0].plot()
                    h, w, ch = result_plotted.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(result_plotted.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap)
                else:
                    # Display original image if no detections
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled_pixmap)
                
                # Update status
                self.status_label.setText("✅ Ready")
                self.progress_bar.setValue(100)
                QApplication.processEvents()
                
                # Hide progress bar
                self.progress_bar.setVisible(False)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error processing image: {str(e)}")
            self.status_label.setText("❌ Error occurred")
            self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for modern look
    window = GrapeDiseaseDetector()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()