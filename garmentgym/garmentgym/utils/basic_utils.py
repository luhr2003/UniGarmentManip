import os
import numpy as np
def get_rotation_matrix(rotationVector, angle):
    angle = float(angle)
    axis = rotationVector/np.sqrt(np.dot(rotationVector , rotationVector))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ]) 

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("warning: dir %s already exists" % path)

def get_rotation_matrix(rotationVector, angle):
    angle = float(angle)
    axis = rotationVector/np.sqrt(np.dot(rotationVector , rotationVector))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ]) 