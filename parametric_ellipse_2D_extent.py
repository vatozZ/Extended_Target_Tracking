import numpy as np
import matplotlib.pyplot as plt

# Given 2D-Xk Matrix
Xk = np.array([[4, 2], [2, 3]])

# Özdeğerlerin ve özvektörlerin hesaplanması
eigenvalues, eigenvectors = np.linalg.eig(Xk)

#öz değerler:
#e1 = 5.5
#e2 = 1.43

#v1 = [0.78, -0.61]
#v2 = [0.61,  0.78]

# Ana eksen ve yedek eksen özvektörlerinin alınması
# Major and minor axis' eigenvectors
major_axis = eigenvectors[:, np.argmax(eigenvalues)]
minor_axis = eigenvectors[:, np.argmin(eigenvalues)]

# eigenvectors.shape = (2, 2)
# eigenvectors = >
# 1st row = 1st eigenvector, 1st eigenvalues
# 2nd row = 1nd eigenvector, 2nd eigenvalues

# center of the ellipse
center = np.array([0, 0])

# Theta values for drawing ellipse
theta = np.linspace(0, 2*np.pi, 100)

minor_length = Xk[0][0]
major_length = Xk[1][1]
if Xk[1][1] < minor_length:
    minor_length = Xk[1][1]
    major_length = Xk[0][0]

#major and minor lengths
a = major_length #1 / np.sqrt(eigenvalues[np.argmax(eigenvalues)])
b = minor_length #1 /  np.sqrt(eigenvalues[np.argmin(eigenvalues)])

x = center[0] + a * np.cos(theta) * major_axis[0] + b * np.sin(theta) * minor_axis[0]
y = center[1] + a * np.cos(theta) * major_axis[1] + b * np.sin(theta) * minor_axis[1]


# Draw Ellipse
plt.figure(figsize=(6, 6))

# Minor Arrow
x_start = 0
y_start = 0
x_end, y_end = eigenvectors[:, np.argmin(eigenvalues)][0], eigenvectors[:, np.argmin(eigenvalues)][1]
x_end = x_end * minor_length
y_end = y_end * minor_length
plt.text(x=x_end, y=y_end+0.1, s = str(round(np.min(eigenvalues), 1)), fontdict={'fontsize': 20})
plt.arrow(x_start, y_start, x_end-x_start, y_end-y_start, head_width=0.2, head_length=0.3, fc='blue', ec='black')

#Major Arrow
x_end, y_end = eigenvectors[:, np.argmax(eigenvalues)][0], eigenvectors[:, np.argmax(eigenvalues)][1]
y_end = y_end * major_length
x_end = x_end * major_length
plt.text(x=x_end, y=y_end+0.1, s=str(round(np.max(eigenvalues), 1)), fontdict={'fontsize': 20})
plt.arrow(x_start, y_start, x_end-x_start, y_end-y_start, head_width=0.2, head_length=0.3, fc='blue', ec='black')

plt.plot(x, y)
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Ellipse from the 2D Matrix')
plt.grid(True)
plt.axis('equal')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.show()