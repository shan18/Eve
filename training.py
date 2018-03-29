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

# Set up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope('optimization'):
    loss_error = tf.contrib.seq2seq.sequence_loss(
        training_predictions,
        targets,
        tf.ones([input_shape[0], sequence_length])  # Initialize the weights by 1
    )
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)

    # Every gradient tensor in the graph also has a variable attached to it.
    # Thus, to apply clipping to all the gradients, two for loops are required,
    # one for the gradient tensor and the other for the variable attached to it.
    clipped_gradients = [
        (tf.clip_by_value(grad_tensor, -5., 5.), grad_variable)
        for grad_tensor, grad_variable in gradients if grad_tensor is not None  # Check if the grad_tensor exists
    ]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
