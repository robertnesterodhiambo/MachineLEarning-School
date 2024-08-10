# src/model.py
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model_path = 'best_skin_disease_model.h5'  # Update this if your model has a different name or path
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
