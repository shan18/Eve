import time
import numpy as np
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


def apply_padding(batch_of_sequences, word2int):
    """ Pad the sequences with <PAD> token

        :example:
        Question: [ 'Who', 'are', 'you' ]
        Answer: [ '<SOS>', 'I', 'am', 'a', 'bot', '.', '<EOS>']

        After padding, the question and the answer becomes of equal length.
        Question (padded): [ 'Who', 'are', 'you' , '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        Answer (padded): [ '<SOS>', 'I', 'am', 'a', 'bot', '.', '<EOS>', '<PAD>']

        :note:
        The padding should be done in such a way that each sentence of a batch has same length.
    """
    max_sequence_length = len(max(batch_of_sequences, key=len))
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


def split_into_batches(questions, answers, batch_size, questions_words2int, answers_words2int):
    """ Split the data into batches of questions and answers.

        :note:
        In order to work with tensorflow, the batches should be in a numpy array.
    """
    for batch_index in range(len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_words2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answers_words2int))
        yield padded_questions_in_batch, padded_answers_in_batch


# Split questions and answers into training and validation sets
training_validation_split = int(len(preprocessed_data['sorted_clean_questions']) * 0.15)
training_questions = preprocessed_data['sorted_clean_questions'][training_validation_split:]
training_answers = preprocessed_data['sorted_clean_answers'][training_validation_split:]
validation_questions = preprocessed_data['sorted_clean_questions'][:training_validation_split]
validation_answers = preprocessed_data['sorted_clean_answers'][:training_validation_split]


# Train the model
training_num_batches = len(training_questions) // batch_size
validation_num_batches = len(validation_questions) // batch_size
batch_index_check_training_loss = 100  # check loss after every 100 batches
batch_index_check_validation_loss = (training_num_batches // 2) - 1  # check loss during half of each epoch
total_training_loss_error = 0  # sum of loss after training 100 batches
list_validation_loss_error = []  # a list is used for early stopping i.e. check if the new loss is minimum of all the previous losses encountered
early_stopping_check = 0  # This will be incremented by 1 every time there is no improvement over the validation loss
early_stopping_stop = 1000  # Stop training when early_stopping_check reaches this value
checkpoint = 'chatbot_weights.ckpt'  # save the weights so that it can be loaded in order to chat with the trained chatbot
session.run(tf.global_variables_initializer())  # Initialize the global variables for the session

for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(
        training_questions, training_answers, batch_size, preprocessed_data['questions_words2int'], preprocessed_data['answers_words2int']
    )):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {
            inputs: padded_questions_in_batch,
            targets: padded_answers_in_batch,
            lr: learning_rate,
            sequence_length: padded_answers_in_batch.shape[1],
            keep_prob: keep_probability
        })
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training loss error: {:>6.3f}, Training Time on {} batches: {:d} seconds'.format(
                epoch,
                epochs,
                batch_index,
                training_num_batches,
                total_training_loss_error / batch_index_check_training_loss,  # Average training loss
                batch_index_check_training_loss,
                int(batch_time * batch_index_check_training_loss)
            ))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:  # exclude first batch
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(
                validation_questions, validation_answers, batch_size, preprocessed_data['questions_words2int'], preprocessed_data['answers_words2int']
            )):
                batch_validation_loss_error = session.run(loss_error, {  # optimizer is not required during validation, thus only one output is returned
                    inputs: padded_questions_in_batch,
                    targets: padded_answers_in_batch,
                    lr: learning_rate,
                    sequence_length: padded_answers_in_batch.shape[1],
                    keep_prob: 1
                })  # no dropout required during validation thus keep_prob is 1
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / validation_num_batches
            print('Validation Loss Error {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                average_validation_loss_error,
                int(batch_time)
            ))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!')
                early_stopping_check = 0
                saver = tf.train.Saver()  # save the model
                saver.save(session, checkpoint)
            else:
                print('Sorry, I do not speak better. Need some more practice!')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print('I cannot speak better than this. This is the best I can do.')
        break
print('Game Over')
