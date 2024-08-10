# src/model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the saved model
model_path = '/home/dragon/Git/model/best_skin_disease_model.h5'
model = load_model(model_path)

# Print the model architecture
print("Model Summary:")
model.summary()

# Print the model layers and their configurations
print("\nModel Layers and Configurations:")
for layer in model.layers:
    print(f"Layer: {layer.name}, Type: {layer.__class__.__name__}")
    print("Configuration:", layer.get_config())
    print("Weights:", layer.get_weights())
    print("-" * 80)

# Print the model's optimizer
print("\nModel Optimizer:")
optimizer_config = model.optimizer.get_config()
for key, value in optimizer_config.items():
    print(f"{key}: {value}")

# Print the model's loss function
print("\nModel Loss Function:")
print(model.loss)

# Print the model's metrics
print("\nModel Metrics:")
print(model.metrics_names)

# Print the total number of parameters in the model
print("\nTotal Parameters:")
print(f"Trainable: {model.count_params()}")

# Directory paths for the validation/test data
data_dir = os.path.expanduser('~/Git/MachineLEarning-School/SkinDiseases/skin-disease-datasaet')
val_dir = os.path.join(data_dir, 'test_set')

# Data preprocessing
val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model on the validation/test set and print accuracy
loss, accuracy = model.evaluate(val_generator)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
