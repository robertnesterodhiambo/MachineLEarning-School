import tensorflow as tf
import numpy as np
from preprocess import get_data, get_next_batch  

class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Define your layers
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')

        # Flatten and Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)  # Dropout layer for regularization
        self.dense2 = tf.keras.layers.Dense(num_classes)  # Final output layer

    def call(self, inputs, is_testing=False):
        # Forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)

        # Flatten the output for dense layer input
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dropout(x)  # Apply dropout only during training

        logits = self.dense2(x)  # Output logits
        
        return logits

def main():
    classes = [3, 4, 5]
    num_classes = len(classes)

    # Load your data
    train_inputs, train_labels = get_data('data/train', classes)
    test_inputs, _ = get_data('data/test', classes)

    # Ensure train_labels are 1D
    train_labels = np.argmax(train_labels, axis=1) if train_labels.ndim > 1 else train_labels

    # Check shapes of the data
    print("train_inputs shape:", train_inputs.shape)
    print("train_labels shape:", train_labels.shape)
    print("Unique train labels:", np.unique(train_labels))

    # Initialize and compile the CNN model
    model = CNN(num_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Train the model
    model.fit(train_inputs, train_labels, batch_size=64, epochs=10, validation_split=0.2)

    # Make predictions on the test set
    predictions = model.predict(test_inputs)

    # Save predictions to a .npy file
    np.save('predictions_cnn.npy', predictions)

if __name__ == "__main__":
    main()
