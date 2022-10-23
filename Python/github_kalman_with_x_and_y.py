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
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    # measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

    fmeasurements = np.array([(0.2, 7), (0.3, 5), (0.2, 6), (0.4, 5)])
    import math
    kx = []
    ky = []
    for m in fmeasurements:
        ky.append(m[1] * math.cos(m[0]))
        kx.append(m[1] * math.sin(m[0]))
    fcoord = np.array(list(zip(kx, ky)))
    #First measurment
    x0 = np.array([[fcoord[0][0], fcoord[0][1]], [0, 0], [0, 0]])

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R, x0 = x0)
    predictions = []

    for z in fcoord:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)
    
    tx, ty = zip(*fcoord)
    print(fcoord)
    px, py = zip(*predictions)
    print(predictions)
    import matplotlib.pyplot as plt
    """plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.legend() """
    plt.plot(px, py, 'go')
    plt.plot(tx, ty, 'ro')
    plt.xlim([-10, 10])
    plt.ylim([0, 10])
    plt.plot(predictions[-1][0], predictions[-1][1], 'bo')
    plt.show()
    print("done")
if __name__ == '__main__':
    example()