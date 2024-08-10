import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    class_mode='categorical',
    shuffle=True
)

# Create the validation data generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

def build_model(hp):
    base_model = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
    base_model.trainable = False  # Freeze the base model weights
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'))
    model.add(BatchNormalization())  # Added BatchNormalization
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

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Perform hyperparameter search
tuner.search(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping, reduce_lr]
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
