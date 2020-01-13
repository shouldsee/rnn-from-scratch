import numpy as np

from preprocessing import getSentenceData
from rnn import Model

word_dim = 100
hidden_dim = 50
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv.1000', word_dim)


np.random.seed(10)
print('Using Autograd')
rnn = Model(word_dim, hidden_dim, use_autograd = True)
losses = rnn.train(X_train[:100], y_train[:100], learning_rate=0.005, nepoch=10, evaluate_loss_after=1)

np.random.seed(10)
print('Using object.backward')
rnn = Model(word_dim, hidden_dim, use_autograd = False)
losses = rnn.train(X_train[:100], y_train[:100], learning_rate=0.005, nepoch=10, evaluate_loss_after=1)
