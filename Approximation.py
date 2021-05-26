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

