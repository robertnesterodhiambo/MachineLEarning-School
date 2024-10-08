from abc import ABC, abstractmethod
import tensorflow as tf

class CifarModel(tf.keras.Model, ABC):
    @abstractmethod
    def call(self, inputs):
        """
        Implement the forward pass of the model.
        :param inputs: Input tensor.
        :return: Output tensor after passing through the model.
        """
        pass

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
                       containing the result of multiple convolution and feed forward layers.
                       Softmax is applied in this function.
        :param labels: during training, a matrix of shape (batch_size, self.num_classes) 
                       containing the train labels.
        :return: the loss of the model as a Tensor.
        """
        # Calculate softmax cross-entropy loss
        loss_value = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        
        # Return the mean loss
        return tf.reduce_mean(loss_value)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, 
                       this will be (batch_size, self.num_classes) containing the result 
                       of multiple convolution and feed forward layers.
        :param labels: matrix of size (batch_size, self.num_classes) containing the answers, 
                       during training, this will be (batch_size, self.num_classes).
        :return: the accuracy of the model as a Tensor.
        """
        # Get the predicted class indices
        predictions = tf.argmax(logits, axis=1)
        
        # Get the correct class indices
        correct_labels = tf.argmax(labels, axis=1)
        
        # Determine if predictions are correct
        correct_predictions = tf.equal(predictions, correct_labels)
        
        # Calculate accuracy
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
