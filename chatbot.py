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
    text = re.sub(r"[-()\"#/@;:<>{}+=-|.?,]", '', text)
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

print(answers_clean[0])
