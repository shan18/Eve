# Importing the necessary packages
import numpy as np
import tensorflow as tf
import re
import time


def clean_text(text):
    """ simplifies the text to make the training process easier """
    text = text.lower()
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"\'ll", ' will', text)
    text = re.sub(r"\'ve", ' have', text)
    text = re.sub(r"\'re", ' are', text)
    text = re.sub(r"\'d", ' would', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"[-()\"#/@;:<>{}+=|.?,]", '', text)
    return text


# Importing the dataset
lines = open('dataset/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Mapping each line to its id in a dictionary
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[-1]

# Create a list of all conversations
conversations_ids = []
for conversation in conversations[:-1]:  # last row is empty
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").split(', ')
    conversations_ids.append(_conversation)

# separating questions and answers from the dataset
questions, answers = [], []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

# clean the questions
questions_clean = []
for question in questions:
    questions_clean.append(clean_text(question))

# clean the answers
answers_clean = []
for answer in answers:
    answers_clean.append(clean_text(answer))

# Map each word to its number of occurances
word2count = {}
for question in questions_clean:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in answers_clean:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# remove the less frequent words based on a threshold and
# map the questions words and answer words to an unique integer
threshold = 20
word_number = 1
questions_words2int = {}
answers_words2int = {}
for word, count in word2count.items():
    if count >= threshold:
        questions_words2int[word] = word_number
        answers_words2int[word] = word_number
        word_number += 1


# Add the last tokens to the dictionaries
# <SOS>: Start of Sentence
# <EOS>: End of Sentence
# <PAD>: Padding, so that the data remains of equal length
# <OUT>: Placeholder tag to replace all the words filtered out by the dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']  # order is important
for token in tokens:
    questions_words2int[token] = len(questions_words2int) + 1
    answers_words2int[token] = len(answers_words2int) + 1

# create inverse mapping for answers_words2int dictionary
answers_int2words = {w_i: w for w, w_i in answers_words2int.items()}

# add <EOS> to end of each answer
for i in range(len(answers_clean)):
    answers_clean[i] += ' <EOS>'

# translate all questions and answers to integers and
# replace filtered out words by <OUT>
questions_to_int = []
for question in questions_clean:
    clean_question = []
    for word in question.split():
        if word in questions_words2int:
            clean_question.append(questions_words2int[word])
        else:
            clean_question.append(questions_words2int['<OUT>'])
    questions_to_int.append(clean_question)
answers_to_int = []
for answer in answers_clean:
    clean_answer = []
    for word in answer.split():
        if word in answers_words2int:
            clean_answer.append(answers_words2int[word])
        else:
            clean_answer.append(answers_words2int['<OUT>'])
    answers_to_int.append(clean_answer)

# Sort questions and answers based on the length of questions
# This improves the training speed by reducing the padding
# Exclude the questions that are too long
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1):  # max length = 25
    for idx, value in enumerate(questions_to_int):
        if len(value) == length:
            sorted_clean_questions.append(questions_to_int[idx])
            sorted_clean_answers.append(answers_to_int[idx])
