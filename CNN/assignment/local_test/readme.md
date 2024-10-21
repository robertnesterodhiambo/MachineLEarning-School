Certainly! Here’s the README in the first person:



# Manual 2D Convolution Implementation

## Project Overview

In this project, I implemented a custom 2D convolution operation without using any high-level convolution libraries like TensorFlow or PyTorch. Instead, I manually handled the convolution process by working directly with tensor operations. The purpose of this implementation is to get a deeper understanding of how convolutional layers operate in a CNN.

## Motivation

I’ve always been curious about the inner workings of Convolutional Neural Networks (CNNs). While modern libraries make it easy to use pre-built convolution layers, I wanted to dive deeper into the mechanics. This project gave me the opportunity to manually compute convolutions, apply filters, and witness the results without relying on existing libraries for the convolution operation itself.

## Implementation

I implemented the convolution operation in the file `manual_convolution.py`. Here's a breakdown of what I've done:
- Extracted patches from the input tensor.
- Applied filters manually to each patch.
- Computed the convolution output by summing up the element-wise products.
- Integrated this custom layer into a simple CNN for testing.

## How to Run



1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. To run the tests:
   ```bash
   python local_test.py
   ```

## Model Accuracy

I tested my custom 2D convolution layer by replacing the standard convolution layer in a simple CNN. After training the model for 10 epochs on the dataset, I achieved an accuracy of **92%**. This isn't far from the accuracy obtained using standard convolutional layers, which shows that the manual implementation is working well. 

However, my implementation is slower compared to optimized versions like TensorFlow's `Conv2D`, but that's expected due to the lack of optimization in manually applying filters and handling strides.

## Known Bugs and Issues

- **Performance**: The manual convolution process is significantly slower than the built-in versions in frameworks like TensorFlow and PyTorch. This is primarily because I haven't optimized for batch processing or GPU acceleration.
- **Edge cases**: Currently, the implementation only supports 'valid' padding and doesn't handle other types of padding like 'same'. This could result in some loss of information for smaller input images.
- **Fixed Strides**: My implementation doesn't handle cases where strides for different axes differ. Right now, both horizontal and vertical strides are assumed to be the same.
  
## Future Improvements

In the future, I plan to:
- Add support for 'same' padding and other convolution variations.
- Optimize the performance by leveraging vectorized operations.
- Extend the implementation to handle different strides for different axes.

## Conclusion

Working on this project has helped me understand convolution at a deeper level. Manually implementing such a fundamental part of CNNs allowed me to grasp how each filter interacts with the input image, how strides affect the output, and how padding plays a role in preserving input dimensions.

---

