import tensorflow as tf

from seq2seq_architecture import model_inputs


# Setting the hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512  # 512 columns in the encoder embedding matrix
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9  # decay of 90%
min_learning_rate = 0.0001  # learning_rate won't go below this
keep_probability = 0.5


# Define a tensorflow session
tf.reset_default_graph()  # reset all the graphs before starting a new session
session = tf.InteractiveSession()

# Load model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Set the sequence length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

# Get the shape of the input tensor
input_shape = tf.shape(inputs)
