import tensorflow as tf

class ClothPredict:

    def __init__(self, learning_rate=1e-4):
        self.learning_rate = learning_rate

    '''Helper functions for new variables'''
    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))


    def new_conv_layer(self,
                       input,              # The previous layer
                       num_input_channels, # Num. channels in prev. layer
                       filter_size,        # Width and height of each filter
                       num_filters,        # Number of filters
                       use_pooling=True):  # Use 2x2 max-pooling

        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=num_filters)

        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += biases

        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        layer = tf.nn.relu(layer)
        return layer, weights

    '''Helper function for flattening layer'''
    def flatten_layer(self, layer):
        # Get shape of the input layer
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    '''Helper function for new fc layer'''
    def new_fc_layer(self, 
                     input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True): # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer


    def forward(self, x_image):
        '''Conv layers 1,2,3'''
        layer_conv1, weights_conv1 = \
            self.new_conv_layer(input=x_image,
                           num_input_channels=3,
                           filter_size=3,
                           num_filters=32,
                           use_pooling=True)


        layer_conv2, weights_conv2 = \
            self.new_conv_layer(input=layer_conv1,
                           num_input_channels=32,
                           filter_size=3,
                           num_filters=32,
                           use_pooling=True)

        layer_conv3, weights_conv3 = \
            self.new_conv_layer(input=layer_conv2,
                           num_input_channels=32,
                           filter_size=3,
                           num_filters=64,
                           use_pooling=True)

        # Flatten layer
        layer_flat, num_features = self.flatten_layer(layer_conv3)

        # FC layer 1
        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=128,
                                 use_relu=True)

        # FC layer 2
        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                 num_inputs=128,
                                 num_outputs=42,
                                 use_relu=False)
        return layer_fc2

    def loss(self,logits, label):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label)
        cost = tf.reduce_mean(cross_entropy)
        return cost

    def optimizer(self, loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)