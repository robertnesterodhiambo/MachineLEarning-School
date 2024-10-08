from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from cnn import CNN  # Assuming you have CNN defined somewhere
from mlp import MLP

import os
import tensorflow as tf
import numpy as np
import random
import math

# Ensures that we run only on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train(model, optimizer, train_inputs, train_labels, batch_size=32):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    :param model: the initialized model to use for the forward and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :param batch_size: number of samples per gradient update
    :return: None
    '''
    # Get the number of training samples
    num_samples = train_inputs.shape[0]
    
    # Create an array of indices and shuffle them
    indices = tf.range(num_samples)
    shuffled_indices = tf.random.shuffle(indices)

    # Shuffle train inputs and labels using the shuffled indices
    train_inputs = tf.gather(train_inputs, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)

    # Iterate over batches
    for i in range(0, num_samples, batch_size):
        # Get the current batch inputs and labels
        batch_inputs = train_inputs[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        # Randomly flip images for data augmentation
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)

        # Perform the forward and backward pass
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model.call(batch_inputs)
            # Compute loss (assuming you have a loss function defined, e.g., cross-entropy)
            loss = tf.keras.losses.categorical_crossentropy(batch_labels, logits, from_logits=True)

        # Compute the gradients of all trainable vars w.r.t loss
        gradients = tape.gradient(loss, model.trainable_variables)

        # Adjust the trainable vars according to the optimizer update rule
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Optionally: print loss or keep track of it
        print(f'Batch {i // batch_size + 1}, Loss: {tf.reduce_mean(loss).numpy()}')

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    num_samples = test_inputs.shape[0]
    total_correct = 0

    # Iterate through test inputs in batches
    batch_size = 32
    for i in range(0, num_samples, batch_size):
        # Get the current batch inputs and labels
        batch_inputs = test_inputs[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]

        # Forward pass
        logits = model.call(batch_inputs)
        predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(batch_labels, axis=1)

        # Count correct predictions
        total_correct += tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.int32))

    # Calculate accuracy
    accuracy = total_correct / num_samples
    print(f'Test Accuracy: {accuracy.numpy():.4f}')
    return accuracy

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_results(image_inputs, logits, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i + 1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title=f"PL: {pl}\nAL: {al}")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(logits, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def main():
    '''
    Read in CIFAR10 data (limited to a subset), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    Students should receive a final accuracy 
    on the testing examples for cat, deer and dog of >=75%.
    
    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = 'data/train'  # Update this path if necessary
    LOCAL_TEST_FILE = 'data/test'      # Update this path if necessary

    # Load your testing and training data using the get_data function
    train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE,3)
    test_inputs, test_labels = get_data(LOCAL_TEST_FILE,3)
    
    input_shape = (32, 32, 3)  # Example input shape for 32x32 RGB images
    num_classes = [3]  # 
    # Initialize your model and optimizer
    model = MLP( num_classes)  # Update input shape and num_classes as needed
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Train your model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train(model, optimizer, train_inputs, train_labels)

        # Test your model
        accuracy = test(model, test_inputs, test_labels)
        print(f'Accuracy after epoch {epoch + 1}: {accuracy.numpy():.4f}')

    # Save your predictions as either "predictions_cnn.npy" or "predictions_mlp.npy"
    # depending on which model you are using
    predictions = model.call(test_inputs)
    np.save('predictions_mlp.npy', predictions.numpy())

    return


if __name__ == '__main__':
    main()
