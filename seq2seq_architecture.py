# Import the necessary packages
import tensorflow as tf


# Create placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # control the dropout rate
    return inputs, targets, learning_rate, keep_prob

# <EOS> is not required for the decoder so it is removed from the targets and
# <SOS> is prepended
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])  # matrix filled with integer representation of <SOS>
    right_side = tf.strided_slice(targets, begin=[0, 0], end=[batch_size, -1], strides=[1, 1])  # extracts subset of targets (excludes the last column)
    preprocessed_targets = tf.concat([left_side, right_side], axis=1)  # Horizontal concat
    return preprocessed_targets

# Encoder RNN layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    """ rnn_size: number of input tensors to the layer
        sequence_length: length of the sequence of questions in each batch """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell(lstm_dropout)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=encoder_cell,
            cell_bw=encoder_cell,
            sequence_length=sequence_length,
            inputs=rnn_inputs,
            dtype=float32
    )
    return encoder_state
