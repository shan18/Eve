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
