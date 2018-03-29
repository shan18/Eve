import tensorflow as tf

from data_preprocess import preprocess_input
from seq2seq_architecture import model_inputs, seq2seq_model


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

# Fetch the preprocessed training data
preprocessed_data = preprocess_input()


# Define a tensorflow session
tf.reset_default_graph()  # reset all the graphs before starting a new session
session = tf.InteractiveSession()

# Load model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Set the sequence length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

# Get the shape of the input tensor
input_shape = tf.shape(inputs)


# Get the training and test predictions
training_predictions, test_predictions = seq2seq_model(
    tf.reverse(inputs, [-1]),  # reverse the dimensions of the inputs tensor
    targets,
    keep_prob,
    batch_size,
    sequence_length,
    len(preprocessed_data['answers_words2int']),
    len(preprocessed_data['questions_words2int']),
    encoding_embedding_size,
    decoding_embedding_size,
    rnn_size,
    num_layers,
    preprocessed_data['questions_words2int']
)
