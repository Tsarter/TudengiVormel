import numpy as np
import math
# https://github.com/zziz/kalman-filter
# https://www.kalmanfilter.net/multiExamples.html

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

    def predict(self):
        self.x = np.dot(self.F, self.x) 
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        # Kalman Gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 
        # THE STATE UPDATE EQUATION 
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        I = np.eye(self.n)
        # THE COVARIANCE UPDATE EQUATION
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        # print(self.x)

def example():
    # standard deviations
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])

    # state transition matrix nx × nx
    F = np.array([[1, 0], [0, 1]])

    # measurement uncertainty nz × nz
    R = np.array([[1, 0], [0, 1]])

    # Observation matrix
    H = np.array([[1, 0], [0, 1]])

    # Test measurments
    measurements = np.array([(0.2, 7), (0.3, 5), (0.2, 6), (0.4, 5), (-0.8, 8), (-0.8, 7), (-0.8, 4)])
    kx = []
    ky = []
    for m in measurements:
        ky.append(m[1] * math.cos(m[0]))
        kx.append(m[1] * math.sin(m[0]))
    locations = zip(kx, ky)
    predictions = []
    x0 = np.array([kx[0], ky[0]]).reshape(2, 1)
    # Set up 
    kf = KalmanFilter(F = F, H = H, R = R, Q = Q, x0 = x0)
    for location in locations:
        location = np.array(location).reshape(2, 1)
        predictions.append(np.dot(H,  kf.predict()))
        kf.update(location)

    px, py = zip(*predictions)
    import matplotlib.pyplot as plt
    plt.plot(px, py, 'ro')
    plt.plot(kx, ky, 'go')
    plt.xlim([-10, 10])
    plt.ylim([0, 10])
    plt.plot(predictions[-1][0], predictions[-1][1], 'bo')
    plt.show()
    print("done")



if __name__ == '__main__':
    example()

