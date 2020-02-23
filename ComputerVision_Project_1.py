
import numpy as np
import cv2
import os
from timeit import default_timer as timer
from skimage.measure import compare_ssim
from statistics import mean

SOURCEPATH = 'multi_plant/'
OUTPUTPATH = 'output/'
COMPAREPATH = 'multi_label/'

TESTING = True

if not os.path.exists ( SOURCEPATH ):
    print (" Path  " + SOURCEPATH + "  does  not  exist !")
    exit (1)
if not os.path.exists ( OUTPUTPATH ):
    print (" Path  " + OUTPUTPATH + "  does  not  exist !")
    exit (1)
if not os.path.exists ( COMPAREPATH ):
    print (" Path  " + COMPAREPATH + "  does  not  exist !")
    exit (1)


def leafSegmentation(cameraID, plantID, dayID, timeID, isTesting):

    # UPLOAD IMAGE AND LABEL
    imageName = '0' + str(cameraID) + '_' + '0' + str(plantID) + '_' + '00' + str(dayID) + '_' + '0' + str(timeID)
    image = cv2.imread( SOURCEPATH + 'rgb_' + imageName + '.png', cv2.IMREAD_COLOR)
    if isTesting:
        cv2.imshow("image1", image )
    label = cv2.imread( COMPAREPATH + 'label_' + imageName + '.png', cv2.IMREAD_COLOR)
    if isTesting:
        cv2.imshow("label", label )

    cv2.putText(image , imageName , (20 ,50) , cv2.FONT_HERSHEY_SIMPLEX , 1, 255)

    # GREEN CHANNEL
    #green = image[:,:,1]
    #if isTesting:
    #    cv2.imshow("green", green )

    # CHANGE FROM BGR TO HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # FIND LEAVES
    lower_green = np.array ([38 , 45, 35])
    upper_green = np.array ([75 , 230 , 170])
    mask = cv2.inRange (hsv, lower_green, upper_green )
    if isTesting:
        cv2.imshow("mask1", mask )

    mask = cv2.medianBlur (mask , 3)
    if isTesting:
        cv2.imshow("medianBlur1", mask )
  
    # MORPHOLOGY
    kernel = np.ones ((3, 3), np.uint8 )
    mask = cv2.morphologyEx (mask , cv2.MORPH_OPEN , kernel, iterations = 2)
    if isTesting:
        cv2.imshow("opening1", mask )

    mask = cv2.medianBlur (mask , 3)
    if isTesting:
        cv2.imshow("medianBlur2", mask )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 2)
    if isTesting:
        cv2.imshow("closing", mask )

    # BIGGEST CONTOUR
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # draw a bounding box arounded the detected plant
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    if isTesting:
        cv2.imshow("box", image)
    #else:
    cv2.imwrite(OUTPUTPATH + 'box_' + imageName + '.png', image)

    # COMPARE
    imageTest = cv2.bitwise_and(image,image,mask = mask)
    imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2GRAY)
    labelTest = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ( temp, imageBin) = cv2.threshold(imageTest, 1, 255, cv2.THRESH_BINARY)
    ( _, labelBin) = cv2.threshold(labelTest, 1, 255, cv2.THRESH_BINARY)

    (score, diff) = compare_ssim(labelBin, imageBin, full=True)
    if isTesting:
        print("Image similarity: ", score)
        cv2.imshow("imageBin", imageBin)
        cv2.imshow("labelBin", labelBin)
        cv2.imshow("diff", diff)

    # COUNT WHITE PIXELS
    intersection = cv2.bitwise_and(imageBin,labelBin)
    intersection_pxl = np.sum(intersection == 255)
    imageBin_pxl = np.sum(imageBin == 255)
    labelBin_pxl = np.sum(labelBin == 255)
    if isTesting:
        cv2.imshow("intersection", intersection)

    # CALCULATE JACCARD INDEX AND DICE COEFFICIENT
    jaccard = intersection_pxl / (imageBin_pxl + labelBin_pxl - intersection_pxl)
    dice = (2 * intersection_pxl) / (imageBin_pxl + labelBin_pxl)    
    if isTesting:
        print("Jaccard: ", jaccard)
        print("Dice: ", dice)

    return score, jaccard, dice;

# BODY

start = timer()

if TESTING:

    camera = 0
    plant = 0
    day = 1
    time = 2

    leafSegmentation(camera, plant, day, time, TESTING)

    end = timer()
    print("Time: ", (end - start))
    cv2.waitKey(0)

else:

    plantAverage = list()
    cameraAverage = list()
    plantDiceAverage = list()
    plantJaccardAverage = list()

    daysData = list()
    jaccardData = list()
    diceData = list()

    for plant in range(5):
        for camera in range(3):
            for day in range(10):
                for time in range(6):
                    print('Processing: ' + '0' + str(camera) + '_' + '0' + str(plant) + '_' + '00' + str(day) + '_' + '0' + str(time))
                    (score,jaccard,dice) = leafSegmentation(camera, plant, day, time, TESTING)
                    daysData.append(score)
                    jaccardData.append(jaccard)
                    diceData.append(dice)
            cameraAverage.append(mean(daysData[60*camera : 60*camera+60]))
        plantAverage.append(mean(daysData[180*plant : 180*plant+180]))
        plantJaccardAverage.append(mean(jaccardData[180*plant : 180*plant+180]))
        plantDiceAverage.append(mean(diceData[180*plant : 180*plant+180]))

    # PRINT DATA
    print("Jaccard Index Average: ", mean(jaccardData))
    print("Plant Jaccard Index Average: ", plantJaccardAverage)
    print("Dice Coefficient Average: ", mean(diceData))
    print("Plant Dice Coefficient Average: ", plantDiceAverage)
    print("Overall Average: ", mean(daysData))
    print("Plant Average: ", plantAverage)
    print("Camera Average: ", cameraAverage)

    # IMAGES WITH BOXES ARE STORED INSIDE OUTPUTPATH

    end = timer()
    print("Time: ", (end - start))
    cv2.waitKey(0)