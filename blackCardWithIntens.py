#!/usr/bin/env python
# coding: utf-8

# In[1]:


from exif import Image #Used for checking image data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os #For file sorting
import cv2 #openCV
import imutils #i forgor
from skimage.draw import line as skiLine #Prevents line clashing with normal line
from scipy.fft import irfft2, fftshift, irfft, ifft2 #IFT imports


# In[2]:


path="images/allBlackCard/blackCard/" #Folder where images are
photos = [] #Array for filepaths
for filename in os.listdir(path):
    my_source = path + filename
    if Image(my_source).has_exif: #Checks for exif data
        my_dest = Image(my_source).datetime_original + ".jpg" #The time taken + .jpg
      
    my_dest = path + my_dest
    os.rename(my_source, my_dest) #Renames the images
    photos.append(my_dest) #Adds the filepath of the image to an array
    
photos.sort() #Sorts the photos by time taken


# In[3]:


seps = []
freq = []
ratio = 14392.0437 #Ratio of pixels per metre
density = 997 #Density of solution
i = 140 #Starting frequency

#Finds the separation values of each pattern, as well as draws a line to visualise each plot
for filename in photos:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (93, 93), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    #Find maxima in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #Centre of central maxima
    M = cv2.moments(cnts[int(len(cnts)/2)]) 
    if M["m00"] != 0:
        startX = int(M["m10"] / M["m00"])
        startY = int(M["m01"] / M["m00"])
    else:
        startX = 0
        startY = 0
    
    #Centre of maxima next to central maxima
    N = cv2.moments(cnts[int(len(cnts)/2) - 1])
    if N["m00"] != 0:
        endX = int(N["m10"] / N["m00"])
        endY = int(N["m01"] / N["m00"])
    else:
        endX = 0
        endY = 0
    
    #Separation using pythag 
    separation = np.sqrt((endY - startY)**2 + (endX - startX)**2)/ratio
    cv2.line(image, (startX, startY), (endX, endY), (0, 255, 0), thickness=2) #Paint line on image
    cv2.imwrite("images/allBlackCard/blackCardLines/" + str(i) + "_Line.jpeg", image) #Save image with the line
    seps.append(separation * 2) #Put separation value into array
    freq.append(i) #Appends frequency to array
    
    i += 10

print(seps)


# In[4]:


length = 2.583
deltaL = 0.001
deltaD = 2/ratio
hVal = 0.15
deltaH = 0.001414213562
laserLambda = 0.000000633
kval = (np.pi * hVal * pd.Series(seps))/(laserLambda * (hVal**2 + length**2))
omega = 2 * np.pi * pd.Series(freq)

deltaKfromL = (pd.Series(seps) * hVal * 2 * length) * (1/((hVal**2 + length**2)**2)) * deltaL #DeltaK from length
deltaKfromD = hVal * (1/(hVal**2 + length**2)) * deltaD #DeltaK from separations
deltaKfromH = (pd.Series(seps)) * (length**2 - hVal**2) *(1/((hVal**2 + length**2)**2)) * deltaH #DeltaK from height calc

deltaK = (np.pi/laserLambda) * (deltaKfromL + deltaKfromD + deltaKfromH) #Uncertainty in k
deltaKCube = 3*(kval ** 2) * deltaK #Uncertainty in k cube

omegasq = omega ** 2
kcube = kval ** 3


# In[5]:


plt.rcParams['font.size'] = 15
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('k³')
ax.set_ylabel('ω²')

def line(x, slope, intercept):
    return slope*x + intercept

# Next few line, fits a line to the (x data, and y data)
popt, pcov = curve_fit(line,kcube,omegasq)
slope = popt[0]
intercept = popt[1]
err_slope = np.sqrt(float(pcov[0][0]))
err_intercept = np.sqrt(float(pcov[1][1]))

print(slope * density * 1000)

ax.scatter(kcube, omegasq, label = "DI Water, Black Card")
ax.errorbar(kcube,           
             omegasq,
             xerr = deltaKCube,
             marker='o',             
             markersize = 4,
             markerfacecolor = 'black',
             color='black',          
             linestyle='none',       
             capsize=6,              
           )

ax.plot(kcube, kcube*slope+intercept, 
         linestyle='--',
         color='black',
         label='Fit to data, m = ' + str(round(slope * density * 1000, 2)))

ax.legend(loc="upper left")


# In[14]:


i = 140
plt.rcParams['font.size'] = 25
#Plots intensity profile and IFT for each pattern
for filename in photos:
    image = cv2.imread(filename) #Pattern
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grayscales the pattern
    blurred = cv2.GaussianBlur(gray, (121, 121), 0) #Applies Gaussian blur
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1] #Thresholds the image
    
    #Finds the maxima in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #Centre of first maxima
    M = cv2.moments(cnts[0]) 
    if M["m00"] != 0:
        startX = int(M["m10"] / M["m00"])
        startY = int(M["m01"] / M["m00"])
    else:
        startX = 0
        startY = 0
    
    #Centre of final maxima
    N = cv2.moments(cnts[int(len(cnts)) - 1])
    if(i == 300): 
        N = cv2.moments(cnts[int(len(cnts)) - 2])
    if N["m00"] != 0:
        endX = int(N["m10"] / N["m00"])
        endY = int(N["m01"] / N["m00"])
    else:
        endX = 0
        endY = 0
    
    #Array of pixels in a line from first to final maxima
    rr, cc = skiLine(startY, startX, endY, endX)
    
    #Intensity value array and a range of pixels to average with
    intens = []
    avgRange = np.linspace(-80, 80, 161)
  
    #Averages the gray value of all pixels within the above range along the line 
    for x in range(0, len(rr)):
        intensValue = 0
        for y in avgRange:
            intensValue += gray[rr[x]][cc[x] + int(y)]/161
        
        intens.append(intensValue)

    #Normalises the intensity values
    intens = intens/np.max(intens)
    
    #Create an array of integer values for the pixel position along the line
    pos = pd.Series(list(range(0, len(intens))))
    
    #Convert pixels to metres
    pos = (pos - len(pos)/2)/ratio
    posTheta = np.arctan(pos/length)
    
    #Perform the IFT on the data
    data = pd.concat([posTheta, np.sqrt(pd.Series(intens))], axis = 1) #For IFT
    ifft = ifft2(data)
    ifft = fftshift(ifft)

    #Create array of IFT without the mirrored plot
    dataList = []
    posIFT = []
    for j in range (0, len(ifft)):
        #if abs(ifft[j][0]) < 0.05:
            dataList.append(-1 * ifft[j][0])
            posIFT.append(posTheta[j])
    
    #Saves plot of IFT to given file
    figIFT = plt.figure(figsize = (10, 10))
    axIFT = figIFT.add_subplot(1, 1, 1)
    axIFT.plot(posIFT, dataList, c = "black")
    axIFT.set_xlabel("Position / m")
    axIFT.set_ylabel("Aperture Function")
    #axIFT.set_xlim(-0.0005, 0.0005)
    #axIFT.legend(loc = "upper left")
    figIFT.savefig("images/allBlackCard/IFT/" + str(i) + "_IFT.png")
    plt.close(figIFT)
    
    #Saves intensity plot to given file
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pos, intens, c = "black")
    ax.set_xlabel("Position / m")
    ax.set_ylabel("Relative Intensity")
    #ax.legend(loc="upper left")
    fig.savefig("images/allBlackCard/intens/" + str(i) + "_intens.png")
    plt.close(fig)
    
    #Paints line onto image and saves to given file
    cv2.line(image, (startX, startY), (endX, endY), (0, 255, 0), thickness=2)
    cv2.imwrite("images/allBlackCard/blackCardFullLines/" + str(i) + "_FullLine.jpeg", image)

    i += 10

print("Done!")


# In[ ]:




