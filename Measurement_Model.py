import math
import numpy as np
from math import pi, cos, sin 
from ETT_constants import *


def generate_measurement_2D():


    #ölçüm matrisi: (n_simülasyon, 3), satır = [x, y, phi]
    measurement_matrix = np.zeros(shape=(n_simulation * numMeasPerInst, 3)) # sütunlar: x, y, phi 

    #Ground Truth matrisi  = (n_simulasyon, 4), satır = [timestamp, centerX, centerY, phi]
    groundTruth_matrix = np.zeros(shape=(n_simulation, 4)) #sütunlar : timestamp, centerX, centerY, phi

    #hedef pozisyonunun başlangıcı
    positionObj = target_initial_position

    #hedefin parametresi = yarıçap(çemberse), tek kenar(karede), uzun eksen, kısa eksen (elipste)
    objParameters = radius

    for i in range(n_simulation):
        
        #oryantasyon hedefin o anki konumuyla sensör arasındaki açı !!!! pi yi anlamadım
        orientationObj = np.pi + np.arctan2(positionObj[1] - sensor_location[1], 
                                            positionObj[0] - sensor_location[0])
        
        
        #ölçüm matrisi ve GT matrisinin ilgili indisleri dolacak.
        current_time = i

        #hız 
        Vx = speed * np.sin(i/(np.pi/3))
        Vy = speed * np.sin(i/(np.pi/3) - np.pi/2)

        velocity = [Vx, Vy]

        velocity = [vel * T for vel in velocity]

        #hedef sabit hızla hareket ediyor.
        positionObj = positionObj + velocity

        current_measurement_array = produceContourMeas(positionObj, orientationObj, numMeasPerInst, objParameters)

        #GT matrisi
        groundTruth_matrix[i] = [current_time, positionObj[0], positionObj[1], orientationObj]

        #ölçüm matrisi
        measurement_matrix[i * numMeasPerInst:(i+1)*numMeasPerInst, 0] = np.ones(numMeasPerInst)*i
        measurement_matrix[i * numMeasPerInst:(i+1)*numMeasPerInst, 1] = current_measurement_array[0]
        measurement_matrix[i * numMeasPerInst:(i+1)*numMeasPerInst, 2] = current_measurement_array[1]
    
    return measurement_matrix, groundTruth_matrix
        

def produceContourMeas(positionObj, orientationObj, numMeasurements, objParameters):

    #hedef parametresi çember için yarıçap
    radius = objParameters

    
    if select_random_theta:
        thetaArray = np.random.uniform(0, 2 * pi , numMeasurements)
    else:
        #gürültülü theta dizisi = (numMeasurement, 1)
        thetaArray = 4 * pi * np.random.randn(numMeasurements) # (row, col) = (numMeasurements, 1)

    #gürültülü mesafe dizisi
    rangeArray = radius + stdMeas * np.random.randn(numMeasurements) #çember merkezinden yarıçap kadar açılıp ölçüm alıyor.

    #polar -> kartezyen koordinat
    xArray, yArray = cart2pol(rangeArray, thetaArray)

    #rotasyon matrisini hesapla
    rotMatrix = np.array([[cos(orientationObj), -sin(orientationObj)],
                          [sin(orientationObj), cos(orientationObj)]])
    
    #pozisyon matrisi
    posMatrix = np.array([xArray, yArray]).reshape(2, -1)

    #pozisyon matrisini döndür.
    posMatrix = np.matmul(rotMatrix, posMatrix)

    #döndürülmüş X-Y koordinat verileri
    xArray = posMatrix[0].reshape(1, -1)
    yArray = posMatrix[1].reshape(1, -1)

    #yeni pozisyon dizileri
    posArray = np.array([xArray + positionObj[0], yArray + positionObj[1]])
    
    return posArray


def cart2pol(rangeArray, thetaArray):

    xArray = np.empty(shape=(rangeArray.shape[0], 1))
    yArray = np.empty(shape=(rangeArray.shape[0], 1))

    for i in range(rangeArray.shape[0]):
        x = rangeArray[i] * cos(np.deg2rad(thetaArray[i]))
        y = rangeArray[i] * sin(np.deg2rad(thetaArray[i]))
        xArray[i] = x
        yArray[i] = y
    
    return xArray, yArray

#Zk, GT = generate_measurement_2D()

