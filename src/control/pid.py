import numpy as np

class PID:
    
    def __init__(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki

    def control(self, x, r, xd, rd, xi, ri):
        return np.array([self.kp*(r-x) + self.kd*(rd-xd) + self.ki*(ri-xi)])