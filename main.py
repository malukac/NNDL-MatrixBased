import network
import mnist_loader 
import numpy as np
import time

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])


t_start = time.time()
#net.SGD(training_data, 1, 5, 1.0, test_data=test_data)
net.SGD(training_data, 30, 10, 0.5, test_data=test_data)
print("Total time: ", time.time() - t_start)
