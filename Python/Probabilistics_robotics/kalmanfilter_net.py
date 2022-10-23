import numpy as np
#https://github.com/zziz/kalman-filter

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
    def predict(self, u = 0):
        # State extrapolation equation
        self.x = np.dot(self.F, self.x)
        # Covariance extrapolation equation
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        # THE STATE UPDATE EQUATION
        # THE COVARIANCE UPDATE EQUATION
        pass

def example():
    # Q -> standard deviations of range, bearing, signature (not Important)
    # Q = np.array([[, 0.0, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    F = np.eye()

    pass

if __name__ == '__main__':
    example()
