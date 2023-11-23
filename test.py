from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define directories for test set
test_dir = './Data/test'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Normalization for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow images from the directory without data augmentation (for test set)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # Assuming it's a multi-class classification
)

# Load the saved model
saved_model = load_model('re_saved_model.h5')

# Evaluate the model on the test data
evaluation = saved_model.evaluate(test_generator)

# Print evaluation metrics (e.g., loss and accuracy)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])
