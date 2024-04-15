"""
This code generates a specified number of samples determined by Poisson distribution,
and uniformly distributed from inside an ellipse defined by its major and minor axes.

@VatozZ
"""

lambda_mean_event = 5
ellips_major_axis = 10 # meter
ellips_minor_axis = 5 # meter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.random import poisson, uniform


total_samples = poisson(lambda_mean_event, 1)[0]

target_ellips_resolution = 10

theta = 0.0

x_ellips = []
y_ellips = []

a = ellips_major_axis
b = ellips_minor_axis

for i in range(0, 360, target_ellips_resolution):

    theta = np.deg2rad(i)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    x_ellips.append(x)
    y_ellips.append(y)

    plt.scatter(x, y, c='red')


# Uniform Sampling Within the Elliptic Region #

def ellips_control(x, y):

    if (x**2 / a**2) + (y**2 / b**2) < 1:
        return True
    else:
        return False

x_ellips = []
y_ellips = []

num_samples_taken = 0

while num_samples_taken < total_samples:

    x = uniform(-a, a)
    y = uniform(-b, b)

    if ellips_control(x, y):
        x_ellips.append(int(x))
        y_ellips.append(int(y))
        num_samples_taken += 1

    else:
        continue

for i in range(0, len(x_ellips)):
    plt.scatter(x_ellips[i], y_ellips[i], c='blue')

blue_patch = mpatches.Patch(color='blue', label='Samples')
red_patch = mpatches.Patch(color='red', label='Target Ellips')

plt.axis('equal')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('n=' + str(total_samples) + ' samples')
plt.legend(handles=[blue_patch, red_patch], loc='best')
plt.grid('on')
#plt.savefig('n_' + str(total_samples) + '_samples.png', dpi=1100, bbox_inches='tight')
plt.show()