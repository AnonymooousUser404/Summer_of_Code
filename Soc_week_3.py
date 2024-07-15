import numpy as np

class Sequential:
    def __init__(self, architecture, loss_function):
        self.architecture = architecture
        self.loss_function = loss_function

    def predict(self, input_data):
        output = input_data
        for layer in self.architecture:
            output = layer.forward(output)
        return output

    def fit(self, X, y,learning_rate):
        
            output = self.predict(X)
            
            grad = self.loss_function(output, y)
            
            for layer in reversed(self.architecture):
                grad = layer.backward(grad, learning_rate)
            
            return output
            

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grad, learning_rate):
        weights_grad = np.dot(self.input.T, grad)
        input_grad = np.dot(grad, self.weights.T)
        self.weights = self.weights - learning_rate * weights_grad
        self.biases = self.biases - learning_rate * grad
        return input_grad

class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, grad, learning_rate):
        sigmoid_grad = self.output * (1 - self.output)
        return grad * sigmoid_grad

def cross_entropy_loss(predictions, labels):
    grad = predictions - labels
    return grad

### ================================================================================================================================================================= ###

# Here I am defining the archectuture list with 2 dense layers and 2 sigmoid layer
architecture = [
    Dense(input_size=2, output_size=5),
    Sigmoid(),
    Dense(input_size=5, output_size=2),
    Sigmoid()
]

model = Sequential(architecture=architecture, loss_function=cross_entropy_loss)

# I am taking random input data 
X = np.random.rand(10,2) 
y = np.array([[0.0],[1.0],[0.0],[0.0],[1.0],[0.0],[0.0],[1.0],[0.0],[1.0]]) # these are just arbitrary
print(X)

# Train the model
prev_output = model.fit(X, y,learning_rate=0.01)
print("Here I am printing the raw output without any changes in the weights and biases of the architcture. \n")
print(prev_output)


# Predict with the model
predictions = model.predict(X)
print("Here I am pringting the predictions of the input data after back propagation(weights and biases are tuned to minimize the loss). \n")
print(predictions)

