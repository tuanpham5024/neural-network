class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # Add a layer to the network
    def add(self, layer):
        self.layers.append(layer)

    def setup_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        """
        :param input: input data [[1, 3]] => 1
        :return: prediction
        """
        result = []
        n = len(input)
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, X_train, y_train, epochs, learning_rate):
        n = len(X_train)
        for i in range(epochs):
            err = 0
            for j in range(n):
                output = X_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # calculate error
                error = self.loss(y_train[j], output)
                err += error

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err = err / n
            print('epoch %d/%d   error=%f' % (i, epochs, err))
