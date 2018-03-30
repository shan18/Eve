import tensorflow as tf


def model_inputs():
    """ Create placeholders for the inputs, targets and some hyperparameters """
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # control the dropout rate
    return inputs, targets, learning_rate, keep_prob


def preprocess_targets(targets, word2int, batch_size):
    """ Preprocess the target data

        :note:
        <EOS> is not required for the decoder, so it is removed from the targets and <SOS> is prepended.
    """
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])  # matrix filled with integer representation of <SOS>
    # extracts a subset of targets (excludes the last column)
    right_side = tf.strided_slice(targets, begin=[0, 0], end=[batch_size, -1], strides=[1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], axis=1)  # Horizontal concat
    return preprocessed_targets


def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    """ Encoder RNN

        :params:
        rnn_inputs: model inputs (tensors returned by the model_inputs() function)
        rnn_size: number of input tensors to the encoder layer
        num_layers: number of layers in the RNN
        keep_prob: control dropout rate to encoder layer
        sequence_length: list of the length of questions in each batch
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)  # add dropout regularization
    # create encoder network consisting of multiple LSTMs
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)

    # make the network bidirectional
    _encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=encoder_cell,
            cell_bw=encoder_cell,
            sequence_length=sequence_length,
            inputs=rnn_inputs,
            dtype=tf.float32
    )
    return encoder_state


def decode_training_set(
        encoder_state, decoder_cell, decoder_embedded_input, sequence_length,
        decoding_scope, output_function, keep_prob, batch_size
):
    """ Training set Decoder
        Step 1: Get the attention states by initializing them to zero
        Step 2: Prepare keys/values/functions for attention
        Step 3: Define a decoder function which will decode the training set

        :params:
        decoder_embedded_input: word embeddings
        decoding_scope: instance of variable_scope class (an advanced data structure) that wraps tensorflow variables
    """
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

    ''' Prepare the attention features
    
        :return:
        attention_states: hidden states to attend over
        attention_option: how to compute attention, either "luong" or "bahdanau"
        num_units: hidden state dimension
        reuse: whether to reuse variable scope
    '''
    attention_keys, attention_values, attention_score_function, attention_construct_function =\
        tf.contrib.seq2seq.prepare_attention(
            attention_states, attention_option='bahdanau', num_units=decoder_cell.output_size
        )

    ''' The attention_decoder_fn_train is a training function for an attention-based sequence-to-sequence model.
        It is used when dynamic_rnn_decoder is in the training mode.
    '''
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(
        encoder_state[0], attention_keys, attention_values, attention_score_function,
        attention_construct_function, name='attn_dec_train'
    )
    decoder_output, _decoder_final_state, _decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell,
        training_decoder_function,
        decoder_embedded_input,
        sequence_length,
        scope=decoding_scope
    )
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# Decode test/validation set
def decode_test_set(
        encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length,
        num_words, decoding_scope, output_function, keep_prob, batch_size
):
    """ Test set Decoder
        Almost same as the Training set Decoder.

        :note:
        There is no need for decoder_embeddings_matrix and sequence_length in dynamic_rnn_decoder and, there is
        also no need for dropout as they are only required during the training.
    """
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function =\
        tf.contrib.seq2seq.prepare_attention(
            attention_states, attention_option='bahdanau', num_units=decoder_cell.output_size
        )
    # Inference is used to logically deduce the output
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
            output_function,
            encoder_state[0],
            attention_keys,
            attention_values,
            attention_score_function,
            attention_construct_function,
            decoder_embeddings_matrix,
            sos_id,
            eos_id,
            maximum_length,
            num_words,
            name='attn_dec_inf'
    )
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoder_cell,
            test_decoder_function,
            scope=decoding_scope
    )
    return test_predictions


def decoder_rnn(
        decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words,
        sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size
):
    """ Decoder RNN

        :note:
        There will be some training required in decoding, thus the need for keep_prob.
    """
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(  # this is the last layer of the network
            x,
            num_words,  # number of outputs
            # activation function is kept the default one i.e. ReLU
            None,  # normalizer
            scope=decoding_scope,
            weights_initializer=weights,
            biases_initializer=biases
        )
        training_predictions = decode_training_set(
            encoder_state,
            decoder_cell,
            decoder_embedded_input,
            sequence_length,
            decoding_scope,
            output_function,
            keep_prob,
            batch_size
        )
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(
            encoder_state,
            decoder_cell,
            decoder_embeddings_matrix,
            word2int['<SOS>'],
            word2int['<EOS>'],
            sequence_length - 1,  # exclude the last token
            num_words,
            decoding_scope,
            output_function,
            keep_prob,
            batch_size
        )
        return training_predictions, test_predictions


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questions_words2int):
    """ Build the sequence to sequence model
        Step 1: Create word embeddings out of the inputs (encoder_embedded_input)
        Step 2: Get the encoder_state from the encoder network
        Step 3: Preprocess the targets
        Step 4: Initialize the decoder_embeddings_matrix
        Step 5: Create word embeddings out of the preprocessed_targets with the help of decoder_embeddings_matrix (decoder_embedded_input)
        Step 6: Get the predictions from the Decoder RNN
    """
    encoder_embedded_input = tf.contrib.layers.embed_sequence(
        inputs,
        vocab_size=answers_num_words + 1,  # +1 because the upper bound of a sequence is always excluded
        embed_dim=encoder_embedding_size,  # number of dimensions in the encoder matrix
        initializer=tf.random_uniform_initializer(0, 1)
    )
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questions_words2int, batch_size)
    
    # initialize decoder embedding matrix with random numbers, it adjusts itself after the training
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform(
        [questions_num_words + 1, decoder_embedding_size], minval=0, maxval=1
    ))  # takes random numbers from a uniform distribution between 0 and 1

    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(
        decoder_embedded_input,
        decoder_embeddings_matrix,
        encoder_state,
        questions_num_words,
        sequence_length,
        rnn_size,
        num_layers,
        questions_words2int,
        keep_prob,
        batch_size
    )
    return training_predictions, test_predictions
