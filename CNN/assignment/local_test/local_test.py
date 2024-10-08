import tensorflow as tf
import numpy as np

from manual_convolution import ManualConv2d
from keras.layers import Conv2D as true_conv2d


def sample_test():  # feel free to remove this
    input_data = np.random.random((1, 10, 10, 3))

    # Define convolutional layer parameters
    filters = 16
    stu_filters = np.zeros((3, 3, 3, 16))
    kernel_size = (3, 3)
    strides = (1, 1)
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="SAME", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)

    # Use true Conv2D layer
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=(10, 10, 3),
        padding="SAME",
        kernel_initializer=tf.keras.initializers.Constant(stu_filters),
    )(input_data)

    # Assert that the outputs are equal
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)
    print("Sample test passed!")


def padding_test_same():
    input_data = np.random.random((1, 4, 4, 1))
    filters = 1
    kernel_size = (3, 3)
    strides = (1, 1)
    stu_filters = np.ones(kernel_size + (input_data.shape[-1], filters))
    # Example from lecture slide
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="SAME", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)
    # test shape
    assert stu_output.shape == input_data.shape

    # test values
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=input_data.shape,
        padding="SAME",
        kernel_initializer=tf.ones,
    )(input_data)
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)
    print("Same padding test passed!")


def padding_test_valid():
    # Example from lecture slide
    input_data = np.random.random((1, 4, 4, 1))
    filters=1
    stu_filters = np.ones((3, 3, 1, 1))
    kernel_size = (3, 3)
    strides = (1, 1)

    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="VALID", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)
    # test shape
    assert stu_output.shape == (1, 2, 2, 1)

    # test values
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=input_data.shape,
        kernel_initializer=tf.ones,
        padding="VALID",
    )(input_data)
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)

    print("Valid padding test passed!")


def base_case_test():
    input_data = np.random.random((1, 1, 1, 3))

    # Define convolutional layer parameters
    filters = 16
    stu_filters = np.zeros((1, 1, 3, 16))
    kernel_size = (1, 1)
    strides = (1, 1)

    # Use your custom conv2d function
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="SAME", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)

    # Use true Conv2D layer
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=(1, 1, 3),
        padding="SAME",
        kernel_initializer=tf.keras.initializers.Constant(stu_filters),
    )(input_data)

    # Assert that the outputs are equal
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)

    print("Base case test passed!")


def weird_shapes_1_same():
    input_data = np.random.random((4, 100, 3, 2))

    # Define convolutional layer parameters
    filters = 16
    stu_filters = np.zeros((3, 2, 2, 16))
    kernel_size = (3, 2)
    strides = (1, 1)

    # Use your custom conv2d function
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="SAME", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)
    # Use true Conv2D layer
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=(100, 3, 2),
        padding="SAME",
        kernel_initializer=tf.keras.initializers.Constant(stu_filters),
    )(input_data)

    # Assert that the outputs are equal
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)

    print("Weird shapes 1 same test passed!")


def weird_shapes_1_valid():
    input_data = np.random.random((4, 100, 3, 2))

    # Define convolutional layer parameters
    filters = 16
    stu_filters = np.zeros((3, 2, 2, 16))
    kernel_size = (3, 2)
    strides = (1, 1)

    # Use your custom conv2d function
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="VALID", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)
    # Use true Conv2D layer
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=(100, 3, 2),
        padding="VALID",
        kernel_initializer=tf.keras.initializers.Constant(stu_filters),
    )(input_data)

    # Assert that the outputs are equal
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)

    print("Weird shapes 1 valid test passed!")


def weird_shapes_2_same():
    input_data = np.random.random((4, 3, 100, 2))

    # Define convolutional layer parameters
    filters = 16
    stu_filters = np.zeros((3, 3, 2, 16))
    kernel_size = (3, 3)
    strides = (1, 1)

    # Use your custom conv2d function
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="SAME", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)
    # Use true Conv2D layer
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=(3, 100, 2),
        padding="SAME",
        kernel_initializer=tf.keras.initializers.Constant(stu_filters),
    )(input_data)

    # Assert that the outputs are equal
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)

    print("Weird shapes 2 same test passed!")


def weird_shapes_2_valid():
    input_data = np.random.random((4, 3, 100, 2))

    # Define convolutional layer parameters
    filters = 16
    stu_filters = np.zeros((2, 3, 2, 16))
    kernel_size = (2, 3)
    strides = (1, 1)

    # Use your custom conv2d function
    stu_conv2d = ManualConv2d(filter_shape=stu_filters.shape, padding="VALID", use_bias=False)
    stu_conv2d.set_weights(stu_filters)
    # stu_conv2d your custom conv2d function
    stu_output = stu_conv2d(input_data)
    # Use true Conv2D layer
    true_output = true_conv2d(
        filters,
        kernel_size,
        strides=strides,
        input_shape=(100, 2, 3),
        padding="VALID",
        kernel_initializer=tf.keras.initializers.Constant(stu_filters),
    )(input_data)

    # Assert that the outputs are equal
    np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)

    print("Weird shapes 2 valid test passed!")


if __name__ == "__main__":
    """
    Uncomment tests to run sanity checks throughout the assignment. These will not be graded
    on gradescope but cover similar edge cases. This way, you can upload to gradescope less frequently.
    """
    ### Simple Tests to Check Conv2D Layers
    sample_test()
    base_case_test()

    ### Tests to Verify Proper Padding Setups
    padding_test_same()
    padding_test_valid()

    ## Tests to Verify Complex Shape Handling
    weird_shapes_1_same()
    weird_shapes_1_valid()
    weird_shapes_2_same()
    weird_shapes_2_valid()
