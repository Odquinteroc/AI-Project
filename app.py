import os
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def delete_all_images():
    # Obtiene la ruta completa a la carpeta de uploads
    upload_folder = app.config['UPLOAD_FOLDER']
    
    # Recorre todos los archivos en la carpeta de uploads
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        
        # Verifica si es un archivo y no una carpeta
        if os.path.isfile(file_path):
            os.remove(file_path)  # Elimina el archivo


# Paths to models
fruit_or_not_model_path = 'models/fruit_or_not.h5'
fruit_type_model_path = 'models/fruit_type.h5'
specific_models_paths = {
    'banana': 'models/Banana.h5',
    #'apple': 'models/Apple.h5',
    #'bellpepper': 'models/Bellpepper.h5',
    'carrot': 'models/Carrot.h5',
    #'cucumber': 'models/Cucumber.h5',
    'grape': 'models/Grape.h5',
    'guava': 'models/Guava.h5',
    'jujube': 'models/Jujube.h5',
    #'mango': 'models/Mango.h5',
    'orange': 'models/Orange.h5',
    'pomegranate': 'models/Pomegranate.h5',
    #'potato': 'models/Potato.h5',
    'strawberry': 'models/Strawberry.h5',
    'tomato': 'models/Tomato.h5'
}

# Define labels for models
fruit_or_not_labels = ["Not Fruit", "Fruit"]
fruit_type_labels = list(specific_models_paths.keys())
print(fruit_type_labels)
healthy_or_rotten_labels = ["Healthy", "Rotten"]

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to preprocess the image from camera
def preprocess_camera_image(image_data):
    image_data = base64.b64decode(image_data.split(',')[1])
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image, cv2.imdecode(np_image, cv2.IMREAD_COLOR)

# Function to save a resized version of the uploaded image
def save_resized_image(image_path, output_path, size=(300, 200)):  # Adjust size here
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, size)
    cv2.imwrite(output_path, resized_image)

# Function to save the image from camera
def save_camera_image(image, filename):
    image_path = os.path.join('uploads', filename)
    cv2.imwrite(image_path, image)
    return image_path

# Load model and predict
def predict(model_path, image, labels):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(image)
    return labels[np.argmax(predictions, axis=1)[0]]

# Generate interpretability image with LIME
def generate_lime_explanation(model, image, filename):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 2 + 0.5, mask)
    
    # Resize the image to 209x180
    img_boundry_resized = cv2.resize(img_boundry, (350, 290))
    
    lime_image_path = os.path.join('uploads', filename)
    #cv2.imwrite(lime_image_path, (img_boundry * 255).astype(np.uint8))
    #return lime_image_path
    cv2.imwrite(lime_image_path, (img_boundry_resized * 255).astype(np.uint8))
    return lime_image_path

# Generate heatmap with LIME
"""
def generate_lime_heatmap(model, image, filename):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    
    # Get the top label for the explanation
    ind = explanation.top_labels[0]
    
    # Extract the heatmap
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.zeros_like(explanation.segments, dtype=np.float32)
    
    for segment, weight in dict_heatmap.items():
        heatmap[explanation.segments == segment] = weight
    
    # Save the heatmap image
    heatmap_image_path = os.path.join('uploads', filename)
    plt.imshow(heatmap, cmap='RdBu', vmin=-np.max(np.abs(heatmap)), vmax=np.max(np.abs(heatmap)))
    plt.colorbar()
    plt.title('LIME Explanation Heatmap')
    plt.axis('off')
    plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return heatmap_image_path
"""
def generate_lime_heatmap(model, image, filename):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    
    # Get the top label for the explanation
    ind = explanation.top_labels[0]
    
    # Extract the heatmap
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.zeros_like(explanation.segments, dtype=np.float32)
    
    for segment, weight in dict_heatmap.items():
        heatmap[explanation.segments == segment] = weight
    
    # Define the size in pixels
    width_px, height_px = 350, 290
    
    # Define DPI (dots per inch). For example, 100 DPI
    dpi = 100
    
    # Calculate the figure size in inches
    width_in = width_px / dpi
    height_in = height_px / dpi
    
    # Create and save the heatmap image
    plt.figure(figsize=(width_in, height_in), dpi=dpi)
    plt.imshow(heatmap, cmap='RdBu', vmin=-np.max(np.abs(heatmap)), vmax=np.max(np.abs(heatmap)))
    plt.colorbar()
    plt.title('LIME Explanation Heatmap')
    plt.axis('off')
    
    heatmap_image_path = os.path.join('uploads', filename)
    plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    
    return heatmap_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict', methods=['POST'])
def predict_image():
    delete_all_images()
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Save a resized version of the uploaded image
    resized_image_path = os.path.join('uploads', 'resized_' + file.filename)
    save_resized_image(file_path, resized_image_path)

    #la imagen que se paso se redimenciona
    image = load_and_preprocess_image(file_path)

    import h5py

    try:
        with h5py.File(fruit_or_not_model_path, 'r') as f:
            print("File opened successfully")
    except OSError as e:
        print(f"Error opening file: {e}")

    # Step 1: Classify if it's a fruit or not
    #llama al modelo que indica si es fruta o no, le pasa la imagen que se ha redimencionado y luego fruit or not labels
    fruit_or_not = predict(fruit_or_not_model_path, image, fruit_or_not_labels)

    #verificar si no 
    print(f"Predicted class: {fruit_or_not}")

    if fruit_or_not == "Not Fruit":
        return jsonify(result="Not a fruit")

    # Step 2: Classify the type of fruit
    fruit_type = predict(fruit_type_model_path, image, fruit_type_labels)

    # Log the fruit type for debugging
    print(f"Predicted fruit type: {fruit_type}")
    print("ya se mostro")

    # Step 3: Classify if the fruit is healthy or rotten
    specific_model_path = specific_models_paths.get(fruit_type, None)
    if specific_model_path is None:
        return jsonify(result=f"Unrecognized fruit type: {fruit_type}")

    healthy_or_rotten = predict(specific_model_path, image, healthy_or_rotten_labels)


    result = f"{fruit_type.capitalize()} is {healthy_or_rotten}"

     # Generate LIME explanation
    specific_model = tf.keras.models.load_model(specific_model_path)
    lime_image_path = generate_lime_explanation(specific_model, image, 'lime_' + file.filename)

    # Generate LIME heatmap explanation
    heatmap_image_path = generate_lime_heatmap(specific_model, image[0], 'heatmap_' + file.filename)

    return jsonify(result=result, original_image_url='/uploads/' + 'resized_' + file.filename, lime_image_url='/uploads/' + 'lime_' + file.filename, heatmap_image_url='/uploads/' + 'heatmap_' + file.filename)
    
    # Generate LIME explanation
    #lime_image_path = generate_lime_explanation(tf.keras.models.load_model(specific_model_path), image, 'lime_' + file.filename)

    #return jsonify(result=result, original_image_url='/uploads/' + 'resized_' + file.filename, lime_image_url='/uploads/' + 'lime_' + file.filename)


@app.route('/predict_camera', methods=['POST'])
def predict_camera_image():
    data = request.get_json()
    image_data = data['image']
    image, original_image = preprocess_camera_image(image_data)

    # Save the original camera image
    file_name = 'camera_image.jpg'
    file_path = save_camera_image(original_image, file_name)
    
    # Save a resized version of the camera image
    resized_image_path = os.path.join('uploads', 'resized_' + file_name)
    save_resized_image(file_path, resized_image_path)

    # Step 1: Classify if it's a fruit or not
    fruit_or_not = predict(fruit_or_not_model_path, image, fruit_or_not_labels)
    if fruit_or_not == "Not Fruit":
        return jsonify(result="Not a fruit")

    # Step 2: Classify the type of fruit
    fruit_type = predict(fruit_type_model_path, image, fruit_type_labels)

    # Log the fruit type for debugging
    print(f"Predicted fruit type: {fruit_type}")

    # Step 3: Classify if the fruit is healthy or rotten
    specific_model_path = specific_models_paths.get(fruit_type, None)
    if specific_model_path is None:
        return jsonify(result=f"Unrecognized fruit type: {fruit_type}")

    healthy_or_rotten = predict(specific_model_path, image, healthy_or_rotten_labels)

    result = f"{fruit_type.capitalize()} is {healthy_or_rotten}"

    # Generate LIME explanation
    lime_image_path = generate_lime_explanation(tf.keras.models.load_model(specific_model_path), image, 'lime_camera_image.jpg')

    return jsonify(result=result, original_image_url='/uploads/' + 'resized_' + file_name, lime_image_url='/uploads/' + 'lime_camera_image.jpg')

if __name__ == '__main__':
    app.run(debug=True, port=5001)