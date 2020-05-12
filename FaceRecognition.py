import glob
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk

images=[]
height = 112
width =92
isMatch = []
accuracy = 0

# Use for UI part
class Application(tk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master, bg= "#E3E5E6")
        self.master = master
        self.grid(sticky = "nesw")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1) 
        
        self.canvas = tk.Canvas(master)
        self.img = []
        
        for i in range(len(images)):
            self.img.append(tk.PhotoImage(file=images[i])) 
        str1 = "Accuracy : "+ str(accuracy) + "%"
        
        # print(str1)
        index = 0
        for i in range(0,len(images),2):
                
                
            matchText = "Match"
            col = "darkblue"
            if(isMatch[index] == 0):
                matchText = "Not Match"
                col = "red"
            self.canvas.create_text(60 , 20,fill="red",font="Calibri",text=str1)
            self.canvas.create_image(0 , index*(height+10)+40, anchor=tk.NW, image=self.img[i])
            self.canvas.create_image(width + 10,index*(height+10)+40, anchor=tk.NW, image=self.img[i+1]) 
            self.canvas.create_text(width*2 + 60 , (height+10)*(index) + height/2+40,fill=col,font="Calibri",text=matchText)
            index += 1

 
        self.canvas.grid(row=0, column=0, sticky = "nesw")

        self.scroll_x = tk.Scrollbar(master, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")

        self.scroll_y = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        self.canvas.configure(scrollregion=(0,0,300,(height+20)*i/2))


    
# adjust brightness and contrast of a given image
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# Image downsampling , increasing contrast and creating a vector
def DownSampling(img): 
    vector = []
    greyScaleImg = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) , dtype = float)

    enhancedImg = np.array(apply_brightness_contrast(greyScaleImg , 100 , -50) , dtype = float )

    downSampledImg = enhancedImg[::5,::5]
    print(np.shape(downSampledImg))
    
    vector = downSampledImg.reshape(1,downSampledImg.size)

    for i in range(len(vector)):
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

    

# add names of the classes to classFolders array
classFolders = []
for root, dirs, files in os.walk("./Training"):
    if(root != "./Training"):
        classFolders.append(root)


# add all the images in "Testing" folder to testImages array
testImages = []
for root, dirs, files in os.walk("./Testing"):
    for name in files:
        if name.endswith((".pgm")):
            testImages.append(name)


XiTs = [] 
# create unique matrix for each class and add to XiTs array(Including class name)
for classFolder in classFolders:
    XiT = getVectors(classFolder+"/*.pgm")
    XiTs.append([XiT,classFolder])

# Uncomment this to see the Unique matrices of each classes.
# for i , j in XiTs:
#     print(i,"\n\n");

Correctcount = 0
Wrongcount = 0

# Get test images one by one and reconizing them. (Using simple linear regression classification algorithm)
for testImage in testImages:
    testImageArr = cv2.imread("./Testing/"+testImage)
    
    minimDistance = 0
    closeClass = ""
    for XiT , classFolder in XiTs:
    
        Xi = XiT.T
        Y = DownSampling(testImageArr)

        h0 = np.dot(XiT,Xi)

        h1 = np.linalg.inv(h0)
        h2 = np.dot(Xi , h1)
        Hi = np.dot(h2,XiT)

        Yi = Hi*Y

        distance = np.linalg.norm(Yi - Y)
        
        if(closeClass == ""):
            minimDistance = distance
            closeClass = classFolder
            
        elif(distance <= minimDistance):
            minimDistance = distance
            closeClass = classFolder
            
    closeImgArr = cv2.imread(closeClass)

    
    a = closeClass.replace("./Training/","")
    b = testImage
    alen = len(a)
    if(a == b[:alen]):
        isMatch.append(1)
        Correctcount = Correctcount +1
    else:
        isMatch.append(0)
        Wrongcount = Wrongcount +1
    # uncomment this to see the each test image name and predicted class name
    # print(b,a)
    
    testImg = "./Testing/"+testImage
    matchImg = closeClass + "/1.pgm"
    images.append(testImg)
    images.append(matchImg)
    

accuracy = (float(Correctcount) /(float(Correctcount) + float(Wrongcount)))*100
# print(accuracy)

root = tk.Tk()
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.geometry("{}x{}+0+0".format(600,400))

app = Application(master=root)
app.mainloop()








