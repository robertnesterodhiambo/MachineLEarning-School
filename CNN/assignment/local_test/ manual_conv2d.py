import tensorflow as tf

class ManualConv2d(tf.Module):
    def __init__(self, filters, kernel_size):
        super(ManualConv2d, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        
        # Initialize the weights of the filters
        self.kernel = tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, filters.shape[-1], filters.shape[0]], stddev=0.1))

    def __call__(self, inputs, strides=[1, 1, 1, 1], padding='SAME'):
        batch_size, in_height, in_width, in_channels = inputs.shape

        # Check that input channels match filter channels
        assert in_channels == self.kernel.shape[2], "Input channels must match filter channels."

        # Calculate padding
        if padding == 'SAME':
            pad_height = (self.kernel_size - 1) // 2
            pad_width = (self.kernel_size - 1) // 2
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_height, pad_height], [pad_width, pad_width], [0, 0]])
        else:
            padded_inputs = inputs

        # Output dimensions
        out_height = (in_height + 2 * pad_height - self.kernel_size) // strides[1] + 1
        out_width = (in_width + 2 * pad_width - self.kernel_size) // strides[2] + 1

        # Initialize output tensor
        output = tf.zeros((batch_size, out_height, out_width, self.filters.shape[0]))

        # Perform convolution
        for b in range(batch_size):  # For each image in the batch
            for h in range(out_height):  # For each output height
                for w in range(out_width):  # For each output width
                    # Calculate the slice of the input that will be convolved
                    h_start = h
                    h_end = h + self.kernel_size
                    w_start = w
                    w_end = w + self.kernel_size
                    
                    # Perform the convolution for each filter
                    for f in range(self.filters.shape[0]):  # For each filter
                        conv_region = padded_inputs[b, h_start:h_end, w_start:w_end, :]
                        output[b, h, w, f] = tf.reduce_sum(conv_region * self.kernel[:, :, :, f])

        return tf.convert_to_tensor(output, dtype=tf.float32)
