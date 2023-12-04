from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import layers, optimizers
import tensorflow as tf

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

# Unfreeze the last few layers of VGG16 for fine-tuning
for layer in base_model.layers[:-4]:
    layer.trainable = True

# Create a new functional model using VGG16 as a base
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(number_of_classes, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
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

# Evaluate the model on the test set
test_metrics = model.evaluate(test_generator)
print("Test Accuracy:", test_metrics[1])
