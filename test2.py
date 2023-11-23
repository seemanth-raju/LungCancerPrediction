from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('re_saved_model.h5')  # Replace with the correct filename

# Path to your new image for testing
new_image_path = './000114 (5).png'  # Replace with the path to your new image

# Load and preprocess the image for prediction
img = keras_image.load_img(new_image_path, target_size=(150, 150))  # Adjust size to match your model input size
img_array = keras_image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
processed_img = img_array / 255.0  # Normalize pixel values

# Make predictions
predictions = loaded_model.predict(processed_img)

# Assuming it's a multi-class classification, you can get class probabilities
print("Predictions:", predictions)
# If it's a classification task, you might interpret the results using class labels or argmax

# For instance, if you have class labels:
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']  # Replace with your class labels
predicted_class = np.argmax(predictions)
predicted_label = class_labels[predicted_class]
print("Predicted Label:", predicted_label)
