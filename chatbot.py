# Importing the necessary packages
import numpy as np
import tensorflow as tf
import re
import time


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
