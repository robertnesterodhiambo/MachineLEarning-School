import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Directory paths
data_dir = os.path.expanduser('~/Git/MachineLEarning-School/SkinDiseases/skin-disease-datasaet')
train_dir = os.path.join(data_dir, 'train_set')
val_dir = os.path.join(data_dir, 'test_set')

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Create the validation data generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('conv1_filters', min_value=32, max_value=128, step=32), 
                     (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(hp.Int('conv2_filters', min_value=64, max_value=128, step=32), 
                     (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(hp.Int('conv3_filters', min_value=128, max_value=256, step=64), 
                     (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.1)))
    model.add(Dense(train_generator.num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Initialize Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=25,
    factor=3,
    directory='my_dir',
    project_name='skin_disease_tuning'
)

# Perform hyperparameter search
tuner.search(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model_path = 'best_skin_disease_model.h5'
best_model.save(best_model_path)

# Print the best hyperparameters
print("Best hyperparameters:", tuner.get_best_hyperparameters(num_trials=1)[0].values)

# Evaluate the best model
results = best_model.evaluate(val_generator)
print(f"Validation Loss: {results[0]}")
print(f"Validation Accuracy: {results[1]}")

# Confirm the best model was saved correctly
if os.path.exists(best_model_path):
    print(f"The best model has been saved to {best_model_path}.")
else:
    print("Error: The best model was not saved.")
