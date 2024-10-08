import tensorflow as tf
import numpy as np

class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape, padding='SAME', use_bias=True):
        """
        Initialize the ManualConv2d layer.

        Parameters:
        - filter_shape: tuple, shape of the convolution filters (height, width, input_channels, output_channels)
        - padding: str, type of padding ('SAME' or 'VALID')
        - use_bias: bool, whether to use bias in the convolution
        """
        super(ManualConv2d, self).__init__()
        self.filter_shape = filter_shape
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the weights for the convolutional filters."""
        self.filters = self.add_weight(name='filters', 
                                       shape=self.filter_shape, 
                                       initializer='glorot_uniform',  # Initialize filters
                                       trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                        shape=(self.filter_shape[-1],), 
                                        initializer='zeros', 
                                        trainable=True)

    def call(self, inputs):
        """Perform the convolution operation."""
        return self.conv2d(inputs, self.filters)

    def conv2d(self, inputs, filters):
        """Perform the manual convolution operation."""
        batch_size, input_height, input_width, input_channels = tf.shape(inputs)

        filter_height, filter_width, _, num_filters = filters.shape

        # Calculate output dimensions
        if self.padding == 'SAME':
            output_height = tf.math.ceil(input_height / 1)
            output_width = tf.math.ceil(input_width / 1)
        elif self.padding == 'VALID':
            output_height = tf.math.ceil((input_height - filter_height + 1) / 1)
            output_width = tf.math.ceil((input_width - filter_width + 1) / 1)

        # Create an output tensor with the correct shape
        output = tf.zeros((batch_size, output_height, output_width, num_filters))

        # Perform convolution
        for b in range(batch_size):
            for k in range(num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i
                        h_end = h_start + filter_height
                        w_start = j
                        w_end = w_start + filter_width

                        # Extract the input patch
                        input_patch = inputs[b, h_start:h_end, w_start:w_end, :]
                        output[b, i, j, k] = tf.reduce_sum(input_patch * filters[:, :, :, k])

                # Add bias if required
                if self.use_bias:
                    output[b, :, :, k] += self.bias[k]

        return output


def sample_test():
    """Sample test function to demonstrate the ManualConv2d layer."""
    # Example filter shape
    filter_shape = (3, 3, 3, 8)  # Height, Width, Input Channels, Output Channels
    stu_conv2d = ManualConv2d(filter_shape)

    # Create a random filter with the correct shape
    stu_filters = np.random.rand(*filter_shape).astype(np.float32)

    # Initialize the layer
    stu_conv2d.build((None, 10, 10, 3))  # Call build to initialize weights
    stu_conv2d.set_weights([stu_filters])  # Pass the filters in a list

    # Create some random input data
    input_data = tf.random.uniform((1, 10, 10, 3))  # Batch size of 1, height and width of 10, 3 channels

    # Call the layer with the input data
    output_data = stu_conv2d(input_data)

    print("Output shape:", output_data.shape)


# Run the sample test if this script is executed
if __name__ == "__main__":
    sample_test()
