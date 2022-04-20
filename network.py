import random
import numpy as np

class Network(object):

  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a)+b)
    return a

  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data: 
      n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      
      x_mini_batches = []
      y_mini_batches = []
      for k in range(0, n, mini_batch_size):
        x_mini_batch = np.empty((self.sizes[0], 0))
        y_mini_batch = np.empty((self.sizes[-1], 0))
        for l in range(k, k + mini_batch_size):
          x_mini_batch = np.hstack((x_mini_batch, training_data[l][0]))
          y_mini_batch = np.hstack((y_mini_batch, training_data[l][1]))
        x_mini_batches.append(x_mini_batch)
        y_mini_batches.append(y_mini_batch)
      mini_batches = list(zip(x_mini_batches, y_mini_batches))
      
      for x, y in mini_batches:
        self.update_mini_batch(x, y, eta)
      if test_data:
        print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
      else:
        print(f"Epoch {j} complete")

  def update_mini_batch(self, x, y, eta):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
   
    nabla_b, nabla_w = self.backprop(x, y)
    self.weights = [w - (eta / x.shape[1]) * nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta / x.shape[1]) * nb for b, nb in zip(self.biases, nabla_b)]

  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
      z = (w @ activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)

    # backward pass
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta @ np.ones((x.shape[1], 1))
    nabla_w[-1] = delta @ activations[-2].transpose()
    
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = (self.weights[-l+1].transpose() @ delta) * sp
      nabla_b[-l] = delta @ np.ones((x.shape[1], 1))
      nabla_w[-l] = delta @ activations[-l-1].transpose()
    return (nabla_b, nabla_w)

  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  def cost_derivative(self, output_activations, y):
    return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))
