import numpy as np


class Approximation(object):

    def __init__(self, x=None, y=None):
        if y is None:
            y = []
        if x is None:
            x = []
        self.x = np.array(x)
        self.y = np.array(y)
        self.flag = 0
        self.A = float
        self.B = float

    def lin(self):
        self.flag = 1
        amount_N = len(self.x)
        amount_X = np.sum(self.x)
        amount_Y = np.sum(self.y)
        amount_XX = np.sum((self.x ** 2))
        amount_XY = np.sum((self.x * self.y))

        det = (amount_XX * amount_N) - (amount_X * amount_X)
        if det != 0:
            self.A = ((amount_XY * amount_N) - (amount_Y * amount_X)) / det
            self.B = ((amount_XX * amount_Y) - (amount_X * amount_XY)) / det
        else:
            print("The system has many solutions")

    def hyp(self):
        self.flag = 2
        amount_N = len(self.x)
        amount_1_xx = np.sum(1 / (self.x ** 2))
        amount_1_x = np.sum(1 / self.x)
        amount_y_x = np.sum(self.y / self.x)
        amount_y = np.sum(self.y)

        det = (amount_1_xx * amount_N) - (amount_1_x * amount_1_x)
        if det != 0:
            self.A = ((amount_y_x * amount_N) - (amount_y * amount_1_x)) / det
            self.B = ((amount_1_xx * amount_y) - (amount_1_x * amount_y_x)) / det
        else:
            print("The system has many solutions")

    def log(self):
        self.flag = 3
        amount_N = len(self.x)
        amount_lnxlnx = np.sum(np.log(self.x) ** 2)
        amount_lnx = np.sum(np.log(self.x))
        amount_ylnx = np.sum(self.y * np.log(self.x))
        amount_y = np.sum(self.y)

        det = (amount_lnxlnx * amount_N) - (amount_lnx * amount_lnx)
        if det != 0:
            self.A = ((amount_ylnx * amount_N) - (amount_y * amount_lnx)) / det
            self.B = ((amount_lnxlnx * amount_y) - (amount_lnx * amount_ylnx)) / det



    def predict(self, x):
        x = np.array(x)
        if self.flag == 1:
            return self.A * x + self.B
        if self.flag == 2:
            return self.A / x + self.B
        if self.flag == 3:
            return self.A * np.log(x) + self.B


class PartyNN(object):

    def __init__(self, learning_rate=0.1, input_nodes=1, hidden_nodes=5, output_nodes=1):
        input_nodes += 1
        self.weights_0_1 = np.random.normal(0.0, hidden_nodes ** -0.5, (hidden_nodes, input_nodes))
        self.weights_1_2 = np.random.normal(0.0, output_nodes ** -0.5, (output_nodes, hidden_nodes))
        self.hyperbolic_tangent_mapper = np.vectorize(self.hyperbolic_tangent)
        self.learning_rate = np.array([learning_rate])

    def set_lr(self, lr):
        self.learning_rate = np.array([lr])

    def hyperbolic_tangent(self, x):
        return np.tanh(x)

    def predict(self, inputs):
        inputs = np.concatenate((inputs, [1]))
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.hyperbolic_tangent_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        # outputs_2 = self.sigmoid_mapper(inputs_2)
        outputs_2 = inputs_2
        return outputs_2

    def train(self, inputs, expected_predict):
        inputs = np.concatenate((inputs, [1]))
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.hyperbolic_tangent_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        # outputs_2 = self.sigmoid_mapper(inputs_2)
        outputs_2 = inputs_2
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = 1  # Здесь косяк# actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = 1 - outputs_1 ** 2
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate
