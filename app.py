import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model, clone_model
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# ---------------------------
# Custom Layers for Model Compatibility
# ---------------------------
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove unsupported 'groups' argument if present.
        kwargs = {k: v for k, v in kwargs.items() if k != 'groups'}
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Remove 'groups' from the config.
        config = {k: v for k, v in config.items() if k != 'groups'}
        return super().from_config(config)

class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    def __init__(self, *args, **kwargs):
        # Remove unsupported keys.
        for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Remove unsupported keys from the config.
        for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
            config.pop(key, None)
        return super().from_config(config)

# ---------------------------
# Paths for Models and Folders
# ---------------------------
ORIGINAL_MODEL_PATH = r"D:\alt+hackj\models\detection_model_video.h5"
UPDATED_MODEL_PATH = r"D:\alt+hackj\models\updated_detection_model_video.h5"

# ---------------------------
# Model Fixing Function
# ---------------------------
def fix_model():
    """
    Load the original model using custom objects,
    remove unsupported parameters via cloning,
    and save the fixed model as UPDATED_MODEL_PATH.
    """
    model = load_model(
        ORIGINAL_MODEL_PATH, 
        compile=False,
        custom_objects={
            'DepthwiseConv2D': CustomDepthwiseConv2D,
            'SeparableConv2D': CustomSeparableConv2D
        }
    )
    
    def clone_fn(layer):
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            config = layer.get_config()
            config.pop('groups', None)
            new_layer = tf.keras.layers.DepthwiseConv2D.from_config(config)
            new_layer.build(layer.input_shape)
            new_layer.set_weights(layer.get_weights())
            return new_layer
        elif isinstance(layer, tf.keras.layers.SeparableConv2D):
            config = layer.get_config()
            for key in ['groups', 'kernel_initializer', 'kernel_regularizer', 'kernel_constraint']:
                config.pop(key, None)
            new_layer = tf.keras.layers.SeparableConv2D.from_config(config)
            new_layer.build(layer.input_shape)
            new_layer.set_weights(layer.get_weights())
            return new_layer
        return layer

    fixed_model = clone_model(model, clone_function=clone_fn)
    fixed_model.set_weights(model.get_weights())
    
    save_model(fixed_model, UPDATED_MODEL_PATH)
    print("Updated model saved to:", UPDATED_MODEL_PATH)
    return fixed_model

# ---------------------------
# Load or Create the Updated Model
# ---------------------------
if not os.path.exists(UPDATED_MODEL_PATH):
    fixed_model = fix_model()  # This will create updated_detection_model_video.h5
else:
    fixed_model = load_model(
        UPDATED_MODEL_PATH, 
        compile=False,
        custom_objects={
            'DepthwiseConv2D': CustomDepthwiseConv2D,
            'SeparableConv2D': CustomSeparableConv2D
        }
    )

# Use the fixed model for deepfake detection.
deepfake_model = fixed_model

# ---------------------------
# Flask App Initialization
# ---------------------------
app = Flask(__name__)
app.secret_key = 'nerain$1'

# Folder configuration
UPLOAD_FOLDER = r'static/uploads'
DATASET_FOLDER = r"dataset"  # Folder with corresponding "reconstructed" images
VIDEO_FOLDER = r"D:\alt+hackj\Result\Real"  # (if processing videos)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_image(frame_bgr):
    """Resize and normalize BGR frame to (1, 256, 256, 3)."""
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_array = image_resized / 255.0
    return np.expand_dims(image_array, axis=0)

# ---------------------------
# Extract Middle Frame from Video
# ---------------------------
def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ---------------------------
# Video Processing: Use Only the Middle Frame
# ---------------------------
def process_video_middle_frame(video_path):
    middle_frame = extract_middle_frame(video_path)
    if middle_frame is None:
        return None, None, None, None

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    original_file = f"{base_name}_middle.jpg"
    original_path = os.path.join(UPLOAD_FOLDER, original_file)
    cv2.imwrite(original_path, middle_frame)
    
    input_data = preprocess_image(middle_frame)
    pred = deepfake_model.predict(input_data)[0][0]

    if pred > 0.5:
        detection_result = f"Deepfake detected (Confidence: {pred * 100:.2f}%)"
        # Look for a corresponding image in the dataset folder.
        dataset_image = os.path.join(DATASET_FOLDER, base_name + '.jpg')
        if os.path.exists(dataset_image):
            reconstructed_file = os.path.basename(dataset_image)
        else:
            reconstructed_file = original_file  # Fallback if not found.
        return detection_result, original_file, reconstructed_file, pred
    else:
        detection_result = f"No deepfake detected (Confidence: {(1 - pred) * 100:.2f}%)"
        return detection_result, original_file, original_file, pred

# ---------------------------
# Process File (Image or Video)
# ---------------------------
def process_file(file_path, file_type):
    base, _ = os.path.splitext(os.path.basename(file_path))
    
    if file_type == "image":
        image = cv2.imread(file_path)
        if image is None:
            return None, None, None
        input_data = preprocess_image(image)
        pred = deepfake_model.predict(input_data)[0][0]
        
        if pred > 0.5:
            detection_result = f"Deepfake detected (Confidence: {pred * 100:.2f}%)"
            # Look for corresponding image in the dataset folder.
            dataset_image_path = os.path.join(DATASET_FOLDER, base + '.jpg')
            if os.path.exists(dataset_image_path):
                reconstructed_file = base + '.jpg'
            else:
                reconstructed_file = ""
        else:
            detection_result = f"No deepfake detected (Confidence: {(1 - pred) * 100:.2f}%)"
            reconstructed_file = os.path.basename(file_path)
        
        original_file = f"{base}_original.jpg"
        original_path = os.path.join(UPLOAD_FOLDER, original_file)
        cv2.imwrite(original_path, image)
        
        return detection_result, original_file, reconstructed_file

    else:
        detection_result, original_file, reconstructed_file, _ = process_video_middle_frame(file_path)
        return detection_result, original_file, reconstructed_file

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        file_type = "image" if file_ext in {'png', 'jpg', 'jpeg', 'gif'} else "video"
        
        detection_result, original_frame, reconstructed_frame = process_file(file_path, file_type)
        if detection_result is None:
            return "Error processing file", 400
        
        return render_template(
            'result.html',
            prediction=detection_result,
            original_frame=original_frame,
            reconstructed_frame=reconstructed_frame
        )
    
    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)
