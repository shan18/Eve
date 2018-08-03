import numpy as np
import tensorflow as tf

from data_preprocess import clean_text, preprocess_input
from seq2seq_architecture import model_inputs, seq2seq_model


def encode_string(text, word2int):
    """ Convert the questions from strings to lists of encoding integers """
    text = clean_text(text)
    return [word2int.get(word, word2int['<OUT>']) for word in text.split()]


''' Fetch the created embeddings '''
preprocessed_data = preprocess_input()
questions_words2int = preprocessed_data['questions_words2int']
answers_words2int = preprocessed_data['answers_words2int']
answers_int2words = preprocessed_data['answers_int2words']


batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
keep_probability = 0.5


''' Store the path to the loaded weights '''
checkpoint = './chatbot_weights.ckpt'


''' Define global tensorflow variables '''
inputs, targets, _, keep_prob = model_inputs()  # Load model inputs
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')  # Set the sequence length


''' Get test predictions '''
_, test_predictions = seq2seq_model(
    tf.reverse(inputs, [-1]),  # reverse the dimensions of the inputs tensor
    targets,
    keep_prob,
    batch_size,
    sequence_length,
    len(answers_words2int),
    len(questions_words2int),
    encoding_embedding_size,
    decoding_embedding_size,
    rnn_size,
    num_layers,
    questions_words2int
)


''' Run the session '''
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)


''' Set up the chat '''
while True:
    question = input('You: ')
    if question.lower() == 'goodbye':
        break
    question = encode_string(question, questions_words2int)
    question += [questions_words2int['<PAD>']] * (20 - len(question))

    # Neural networks accept only batches of input, so we add the question to a dummy batch
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = question

    predicted_answer = session.run(test_predictions, {
        inputs: fake_batch, keep_prob: keep_probability
    })[0]  # We only need the first element of the list

    # Clean the answer
    answer = []
    for i in np.argmax(predicted_answer, axis=1):  # return the token ids of the predicted_answer
        if answers_int2words[i] == 'i':
            token = 'I'
        elif answers_int2words[i] == '<EOS>':
            token = '.'
        elif answers_int2words[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answers_int2words[i]
        answer += token

        if token == '.':  # '.' means that the chatbot has finished the answer. So we stop the loop
            break

    print('Chatbot: ' + answer)
