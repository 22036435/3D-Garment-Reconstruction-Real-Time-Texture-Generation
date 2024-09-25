import sys
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog, QSplitter, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Texture Generator and Viewer")
        self.setGeometry(100, 100, 1600, 900)  # Larger window size to fit both previews

        self.model = self.load_stable_diffusion_model()
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Horizontal layout for texture and model preview with resizable splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left side layout for texture generation and preview
        left_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Input field for texture prompt
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText("Enter texture prompt")
        left_layout.addWidget(self.prompt_input)

        # Generate texture button
        self.generate_button = QPushButton("Generate Texture", self)
        self.generate_button.clicked.connect(self.generate_texture)
        left_layout.addWidget(self.generate_button)

        # Scrollable texture preview label
        texture_scroll_area = QScrollArea(self)
        self.texture_preview_label = QLabel(self)
        texture_scroll_area.setWidgetResizable(True)
        texture_scroll_area.setWidget(self.texture_preview_label)
        left_layout.addWidget(texture_scroll_area)

        # Load FBX button
        self.load_fbx_button = QPushButton("Load FBX File", self)
        self.load_fbx_button.clicked.connect(self.load_fbx)
        left_layout.addWidget(self.load_fbx_button)

        # Apply texture button
        self.apply_texture_button = QPushButton("Apply Texture", self)
        self.apply_texture_button.clicked.connect(self.apply_texture)
        self.apply_texture_button.setEnabled(False)  # Disabled until texture is generated
        left_layout.addWidget(self.apply_texture_button)

        # Right side layout for 3D model preview (rendered model from Blender)
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # Scrollable model preview label
        model_scroll_area = QScrollArea(self)
        self.model_preview_label = QLabel(self)
        model_scroll_area.setWidgetResizable(True)
        model_scroll_area.setWidget(self.model_preview_label)
        right_layout.addWidget(model_scroll_area)

        # Add the left and right layouts to the splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Add splitter to the main layout
        main_layout.addWidget(splitter)

        # Set main layout to the window
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Paths
        self.fbx_path = None
        self.texture_path = "/home/aslico/Desktop/generated_texture.png"  # Path for generated texture
        self.output_fbx_path = "/home/aslico/Desktop/processed_model.fbx"  # Output model path (from Blender)
        self.output_image_path = "/home/aslico/Desktop/processed_model_preview.png"  # Path for rendered preview

    def load_stable_diffusion_model(self):
        return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def generate_texture(self):
        prompt = self.prompt_input.text()
        print(f"Prompt entered: {prompt}")
        texture = self.generate_texture_with_sd(prompt)
        if texture:
            if self.save_texture(texture, self.texture_path):
                print("Texture generated and saved.")
                self.display_texture_preview(self.texture_path)
                self.apply_texture_button.setEnabled(True)  # Enable the apply button after texture generation
            else:
                print("Failed to save texture.")
        else:
            print("Texture generation failed.")

    def generate_texture_with_sd(self, prompt):
        with torch.no_grad():
            generated_images = self.model(prompt).images
        return generated_images[0]

    def save_texture(self, texture, output_path):
        try:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            texture.save(output_path)
            print(f"Texture saved at: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving texture: {e}")
            return False

    def display_texture_preview(self, texture_path):
        # Display the generated texture in the QLabel for review
        pixmap = QPixmap(texture_path)
        self.texture_preview_label.setPixmap(pixmap)
        self.texture_preview_label.setScaledContents(True)

    def load_fbx(self):
        # Open file dialog to choose an FBX file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Load FBX File", "", "FBX Files (*.fbx);;All Files (*)", options=options)
        if file_path:
            self.fbx_path = file_path
            print(f"FBX model loaded: {self.fbx_path}")

    def apply_texture(self):
        if self.fbx_path and os.path.exists(self.texture_path):
            print("Running Blender to apply texture...")
            blender_path = "/usr/bin/blender"  # Path to Blender executable
            blender_script = "blender_script.py"  # Path to the Blender script

            # Call Blender in the background to apply the texture and export the model and preview image
            try:
                subprocess.run([blender_path, '--background', '--python', blender_script, '--', self.fbx_path, self.texture_path, self.output_fbx_path, self.output_image_path])
                print(f"Processed model saved at: {self.output_fbx_path}")
                print(f"Preview image saved at: {self.output_image_path}")

                # Load the rendered image preview in the QLabel
                self.display_rendered_image(self.output_image_path)

            except Exception as e:
                print(f"Error running Blender script: {e}")

    def display_rendered_image(self, image_path):
        # Display the rendered model preview image in the QLabel
        pixmap = QPixmap(image_path)
        self.model_preview_label.setPixmap(pixmap)
        self.model_preview_label.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
