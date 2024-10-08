import pickle
import numpy as np
import tensorflow as tf
import os


def unpickle(file) -> dict[str, np.ndarray]:
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.
    NOTE: DO NOT EDIT
    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_next_batch(idx, inputs, labels, batch_size=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an index, returns the next batch of data and labels. Ex. if batch_size is 5, 
    the data will be a numpy matrix of size 5 * 32 * 32 * 3, and the labels returned will be a numpy matrix of size 5 * 10.
    """
    return (inputs[idx*batch_size:(idx+1)*batch_size], np.array(labels[idx*batch_size:(idx+1)*batch_size]))


def get_data(file_path, classes) -> tuple[np.ndarray, tf.Tensor]:
    """
    Given a file path and a list of class indices, returns an array of 
    normalized inputs (images) and an array of labels. 
    
    - **Note** that because you are using tf.one_hot() for your labels, your
    labels will be a Tensor, hence the mixed output typing for this function. This 
    is fine because TensorFlow also works with NumPy arrays, which you will
    see more of in the next assignment. 

    :param file_path: file path for inputs and labels, something 
                      like 'CIFAR_data_compressed/train'
    :param classes: list of class labels (0-9) to include in the dataset

    :return: normalized NumPy array of inputs and tensor of labels, where 
             inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and 
             Tensor of labels with size (num_examples, num_classes)
    """
    # Load data from the pickle file
    unpickled_file: dict[str, np.ndarray] = unpickle(file_path)
    inputs: np.ndarray = np.array(unpickled_file[b'data'])  # shape (num_samples, 32*32*3)
    labels: np.ndarray = np.array(unpickled_file[b'labels'])  # shape (num_samples,)

    # Filter inputs and labels to only include the specified classes
    mask = np.isin(labels, classes)  # Create a mask for filtering desired classes
    filtered_inputs = inputs[mask]   # Apply mask to filter inputs
    filtered_labels = labels[mask]   # Apply mask to filter labels

    # Reshape the inputs to (num_samples, 3, 32, 32) first
    filtered_inputs = filtered_inputs.reshape(-1, 3, 32, 32)  # Reshape to (num_samples, 3, 32, 32)

    # Normalize the input images (scale pixel values from 0-255 to 0-1)
    normalized_inputs = filtered_inputs.astype(np.float32) / 255.0

    # Transpose to get the shape (num_samples, 32, 32, 3)
    normalized_inputs = np.transpose(normalized_inputs, (0, 2, 3, 1))  # Transpose to (num_samples, 32, 32, 3)

    # One-hot encode the labels using tf.one_hot
    one_hot_labels = tf.one_hot(filtered_labels, depth=len(classes))

    # Return the filtered and normalized inputs and labels
    return normalized_inputs, one_hot_labels

# Example usage of get_data (this would be part of your main training/testing script)
if __name__ == "__main__":
    # Path to the dataset
    train_file_path = "data/train"
    test_file_path = "data/test"
    
    # Specify the classes you're interested in (e.g., 3 = cat, 4 = deer, 5 = dog)
    target_classes = [3, 4, 5]
    
    # Get the training data for the specified classes
    train_inputs, train_labels = get_data(train_file_path, target_classes)
    
    # Get the test data for the specified classes
    test_inputs, test_labels = get_data(test_file_path, target_classes)

    print("Training data shape:", train_inputs.shape)
    print("Training labels shape:", train_labels.shape)
    print("Test data shape:", test_inputs.shape)
    print("Test labels shape:", test_labels.shape)
