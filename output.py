from autograd import numpy as np
# import numpy as np

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self, x, y):
        probs = self.predict(x)
        target = np.zeros((len(probs)),'float32')
        target[y] = 1.
        probs = probs - target
        # probs[y] -= 1.0
        return probs