from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from any origin, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your allowed origins if known
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load_model('transfer_learning_model.h5')

# Define class labels (modify according to your model's output)
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']# Replace with your actual class labels

# Define the function to preprocess the uploaded image
def preprocess_image(file_content):
    img = image.load_img(BytesIO(file_content), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file content
    file_content = await file.read()

    # Preprocess the image
    processed_img = preprocess_image(file_content)

    # Get predictions
    prediction = model.predict(processed_img).flatten()

    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    
    # Get the class name with the highest probability
    predicted_class = class_labels[predicted_class_index]

    return {"predicted_class": predicted_class}
