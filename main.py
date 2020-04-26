import glob
from PIL import Image , ImageOps
import numpy as np 
import cv2


def DownSampling(img): 
    vector = []
    # Convert RGB image to Grey-scale
    greyScaleArr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) , dtype = float)
    # Down sampling (meka hithala gahapu ekk)
    downSampledArr = greyScaleArr[::5,::5]
    # convert 2D matrix to row matrix
    vector = downSampledArr.reshape(1,downSampledArr.size)
    # maxVal = vector.max()
    # minVal = vector.min()
    
    # mapping between 0-1
    for i in range(len(vector)):
        # vector[i] = (vector[i] - minVal) / (maxVal - minVal)
        vector[i] = vector[i] / 255.0
    return vector

# when training file is given return vectors(Xi)
def getVectors(file_path):
    images = [cv2.imread(file) for file in glob.glob(file_path)]
    vectors = np.array([])
    for img in images:
        vector = DownSampling(img)
        
        if(vectors.size == 0):
            vectors = vector
        else:
            vectors = np.append(vectors,vector,axis=0)

    return vectors



# Testing
XiT = getVectors("s1/*.jpg")
Xi = XiT.T


test_img = [cv2.imread(file) for file in glob.glob("./Testing/*.jpg")]
Y = DownSampling(test_img[0])

# Hi = np.dot(np.dot(Xi , np.linalg.inv(np.dot(XiT,Xi))),XiT)
h0 = np.dot(XiT,Xi)


h1 = np.linalg.inv(h0)
h2 = np.dot(Xi , h1)
Hi = np.dot(h2,XiT)

Yi = Hi*Y

distance = np.linalg.norm(Yi - Y)

print(distance)




