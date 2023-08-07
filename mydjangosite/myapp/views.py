from django.shortcuts import render
import pickle
import os
import numpy as np

import tensorflow as tf
import torch

from torch import nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embedding_dim)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            64,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.8
        )

        self.linear = nn.Linear(256, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        hidden, _ = self.lstm(x)
        avg_pool = torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.out(self.linear(out))

        return out


def index(request):
    estimate = ""
    predict = ""
    if request.method == 'POST':
        message = request.POST['message']
        print(message)

        tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.pickle')
        embedding_matrix_path = os.path.join(os.path.dirname(__file__), 'embedding_matrix.pickle')
        model_path = os.path.join(os.path.dirname(__file__), 'movie_review_LSTM.pth')

        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(embedding_matrix_path, 'rb') as handle:
            embedding_matrix = pickle.load(handle)

        model_glove = LSTM(embedding_matrix)
        model_glove.load_state_dict(torch.load(model_path))

        sentence = [message]
        sentence_token = tokenizer.texts_to_sequences(sentence)
        sentence_token = tf.keras.preprocessing.sequence.pad_sequences(sentence_token, maxlen=128)
        sentence_train = torch.tensor(sentence_token, dtype=torch.long)
        predict = int(np.ceil(model_glove(sentence_train).item() * 10))
        if predict > 0.5:
            estimate = 'positive'
        else:
            estimate = 'negative'

    return render(request, "myapp/index.html", {'pred': predict, 'estimate': estimate})
