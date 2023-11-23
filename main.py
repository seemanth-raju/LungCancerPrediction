from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Define directories for train, validation, and test sets
train_dir = './Data/train'
validation_dir = './Data/valid'
test_dir = './Data/test'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Data augmentation and normalization for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# Normalization for validation and test data (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow images from directories and perform data augmentation (for training set)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

# Flow images from directories without data augmentation (for validation and test sets)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

number_of_classes = 4

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0007),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with fine-tuning
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size)

# Save the model for future use (transfer learning model based on VGG16)
model.save('transfer_learning_model.h5')
