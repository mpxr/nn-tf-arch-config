import config_reader
import tensorflow as tf


def get_weights_and_biases(input_variable_size, layers, neurons):
    weights = []
    biases = []
    for current_layer in range(layers):
        layer_name = 'weights' + str(current_layer + 1)
        bias_name = 'bias' + str(current_layer + 1)

        if current_layer == 0:  # first hidden layer
            layer_i = tf.Variable(tf.random_normal([input_variable_size, neurons[current_layer]]),
                                  name=layer_name)
            bias_i = tf.Variable(tf.random_normal([neurons[current_layer]]), name=bias_name)

        elif current_layer < layers - 1:  # middle hidden layers
            layer_i = tf.Variable(tf.random_normal([neurons[current_layer], neurons[current_layer + 1]]),
                                  name=layer_name)
            bias_i = tf.Variable(tf.random_normal([neurons[current_layer]]), name=bias_name)

        else:  # last hidden layer
            layer_i = tf.Variable(tf.random_normal([neurons[current_layer], 1]), name=layer_name)
            bias_i = tf.Variable(tf.random_normal([1]), name=bias_name)

        weights.append(layer_i)
        biases.append(bias_i)

    return weights, biases


def create_fully_connected_architecture(input_variable_size, weights, biases, layer_size):
    if len(weights) != len(biases) != layer_size:
        raise Exception('Weights and bias list have different size!')

    x = tf.placeholder(tf.float32, [None, input_variable_size], name="x")

    layer_1 = tf.add(tf.matmul(x, weights[0]), biases[0])
    layer_1_relu = tf.nn.relu(layer_1)

    previous_layer = layer_1_relu

    for layer in range(1, layer_size - 1):
        layer_i = tf.add(tf.matmul(previous_layer, weights[layer]), biases[layer])
        layer_i_relu = tf.nn.relu(layer_i)
        previous_layer = layer_i_relu

    out_layer = tf.add(tf.matmul(previous_layer, weights[layer_size - 1]), biases[layer_size - 1], name="out_layer")

    return out_layer


def get_architecture(config):
    layer_size = config["layers"]
    neurons = config["neurons"]
    input_variable_size = config["input_variable"]

    weights, biases = get_weights_and_biases(input_variable_size, layer_size, neurons)
    out = create_fully_connected_architecture(input_variable_size, weights, biases, layer_size)

    print(weights)
    print(biases)


if __name__ == "__main__":
    configurations = config_reader.read(path='../config/arch.json')

    for current_config in configurations:
        get_architecture(current_config)
