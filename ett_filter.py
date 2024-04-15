import numpy as np
import matplotlib.pyplot as plt
from numpy import array, exp, matmul, kron
from ett_constants import T, tao, radius,accel_width, sensor_location, theta_maneuver
from numpy.linalg import pinv
import matplotlib.patches as mpatches
import shapely.geometry as sg

def predict_ETT(xk, Pk, vk, Xk, d=2):

    Fk = array([[1., T, T** 2 / 2],
                [0., 1, T],
                [0, 0, exp(-T / theta_maneuver)]])
    Id = np.eye(d)  # (2, 2)
    Phi_k = np.kron(Fk, Id)
    acc_I = np.zeros(shape=(3, 3))
    acc_I[-1][-1] = 1.0
    Dk = accel_width ** 2 * (1 - exp(-2 * T / theta_maneuver)) * acc_I

    #hedef kinematiği güncellemesi

    xk = matmul(Phi_k, xk)
    
    Pk = matmul(matmul(Fk, Pk), Fk.T) + Dk
    
    #hedef uzantı kısmı
    vk = exp(-T/tao) * vk
    
    Xk = (exp(-T/tao) * vk - d -1) / (vk - d - 1) * Xk

    return xk, Pk, vk, Xk

def update_ETT(xk, Pk, vk, Xk, zk, Zk, nk, d=2, s=3):

    Id = np.eye(d)  # (2, 2)
    Is = np.eye(s) # (3, 3)
    zk = zk.reshape(2, 1)

    Hk = array([1, 0, 0]).reshape(1, s)

    # Update Kinematics #
    Sk = Hk @ Pk @ Hk.T + (1/nk)
    
    Wk = Pk @ Hk.T @ pinv(Sk)

    xk = xk + kron(Wk, Id) @ (zk - kron(Hk, Id) @ xk)
    Pk = Pk - Wk @ Sk @ Wk.T

    # Update Extension #
    vk = vk + nk

    Nk = pinv(Sk) * (zk - kron(Hk, Id) @ xk) @ np.transpose(zk - kron(Hk, Id) @ xk)
    
    Xk = Xk + Nk + Zk

    return xk, Pk, Xk, vk

def calculate_mean_and_covariance(Zk, d=2):

    #ölçüm merkezi
    nk = len(Zk) # 10
    
    centroid = np.sum(Zk, axis=0) / nk #mean centroid vector
    
    SM = np.zeros(shape=(d,d)) #scattering covariance matrix
    
    for i in range(nk):
        zk_j = Zk[i]
        
        dif_x = zk_j[0] - centroid[0]
        dif_y = zk_j[1] - centroid[1]

        SM[0][0] = SM[0][0] + dif_x * dif_x
        SM[1][0] = SM[1][0] + dif_x * dif_y
        SM[0][1] = SM[0][1] + dif_y * dif_x
        SM[1][1] = SM[1][1] + dif_y * dif_y

    SM = SM / nk
    
    return centroid, SM, nk
    
def plot_filter(xk, Xk,  Zk, i, ground_truth_matrix):
    
    px, py = xk[:2]

    # Matrisin özdeğerlerini ve özvektörlerini bul
    eigenvalues, eigenvectors = np.linalg.eig(Xk)

    # çemberin merkezini ve yarı eksen uzunluklarını hesapla
    center = [px, py]
    semi_axes_lengths = np.sqrt(eigenvalues)

    # çemberin çizim için theta değerlerini oluştur
    theta = np.linspace(0, 2*np.pi, 100)

    # çemberin x ve y koordinatlarını hesapla
    circle_x = center[0] + semi_axes_lengths[0] * np.cos(theta) * eigenvectors[0, 0] + semi_axes_lengths[1] * np.sin(theta) * eigenvectors[0, 1]
    circle_y = center[1] + semi_axes_lengths[0] * np.cos(theta) * eigenvectors[1, 0] + semi_axes_lengths[1] * np.sin(theta) * eigenvectors[1, 1]

    # çemberi çizdir
    plt.plot(circle_x, circle_y, c='blue')
    # çember merkezini işaretle
    plt.scatter(center[0], center[1], color='blue', marker='x')
    # Ölçümleri çizdir
    for j in Zk:
        plt.scatter(j[0], j[1], c='green', marker='o')
    #Ground Truth çizdir
    plt.scatter(ground_truth_matrix[i][1], ground_truth_matrix[i][2], marker='x', color='black')
    #Ground truth çemberi
    axis = plt.gca()
    circle_obj = plt.Circle((ground_truth_matrix[i][1], ground_truth_matrix[i][2]), radius, fill=False, label='GT')
    axis.add_artist(circle_obj)
    
    #sensor location
    plt.scatter(sensor_location[0], sensor_location[1], color='red', marker='o', s=100)
    
    blue_patch = mpatches.Patch(color='blue', label='Xk')
    black_patch = mpatches.Patch(color='black', label='GT')
    green_patch = mpatches.Patch(color='green', label='Zk')
    red_patch = mpatches.Patch(color='red', label='Sensor')
    plt.legend(handles=[blue_patch, black_patch, green_patch, red_patch], loc='upper right')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    #plt.pause(1)

def get_Zk(n_sim, measurement_matrix):

    meas_X, meas_Y = [], []

    for i in range(measurement_matrix.shape[0]):
        if measurement_matrix[i][0] == n_sim:
            meas_X.append(measurement_matrix[i][1])
            meas_Y.append(measurement_matrix[i][2])
    
    Zk = list(zip(meas_X, meas_Y))

    return Zk

def calculate_error(xk, Xk, ZK, i, ground_truth_matrix):

    eigenvalues, eigenvectors = np.linalg.eig(Xk)

    px, py = xk[:2]

    # çemberin merkezini ve yarı eksen uzunluklarını hesapla
    center = [px, py]
    semi_axes_lengths = np.sqrt(eigenvalues)

    # çemberin çizim için theta değerlerini oluştur
    theta = np.linspace(0, 2 * np.pi, 100)

    # çemberin x ve y koordinatlarını hesapla
    circle_x = center[0] + semi_axes_lengths[0] * np.cos(theta) * eigenvectors[0, 0] + semi_axes_lengths[1] * np.sin(
        theta) * eigenvectors[0, 1]
    circle_y = center[1] + semi_axes_lengths[0] * np.cos(theta) * eigenvectors[1, 0] + semi_axes_lengths[1] * np.sin(
        theta) * eigenvectors[1, 1]

    #plt.plot(circle_x, circle_y, c='blue')

    gt_circle = sg.Point(ground_truth_matrix[i][1], ground_truth_matrix[i][2]).buffer(radius)

    #gt_circle = list(circle.exterior.coords)

    extend = sg.Polygon(list([*zip(circle_x, circle_y)]))

    intersection_of_union = gt_circle.intersection(extend).area

    #plt.plot(extend.exterior.xy[0], extend.exterior.xy[1])

    return intersection_of_union



