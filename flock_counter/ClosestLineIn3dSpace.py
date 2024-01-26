from display3d import show as show3d
import random
import numpy as np


# circles = np.array([
#     [random.randint(2, 8) + n for n in range (0, 100, 5)]
#     for _ in range(3)
# ])

# A = np.column_stack((circles, np.ones(len(circles))))
# coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, np.zeros(len(circles)))
# A, B, C, D = coefficients

# # Set z = 0
# point_on_line_z0 = np.array([
#     0,  # x
#     0,  # y
#     -D/C  # z
# ])

# # Set y = 0
# point_on_line_y0 = np.array([
#     0,  # x
#     -D/B,  # y
#     0  # z
# ])

# print(point_on_line_y0, point_on_line_z0)
# show3d(circles=list(circles), lines=[[],[],[]])

import numpy as np

# Generate some data that lies along a line

x = np.mgrid[-2:5:120j] * 187
y = np.mgrid[1:9:120j] * 69
z = np.mgrid[-5:3:120j] * 420


min_x = np.min(x)
min_y = np.min(y)
min_z = np.min(z)

max_x = np.max(x)
max_y = np.max(y)
max_z = np.max(z)

a = max_x - min_x
b = max_y - min_y
c = max_z - min_z

d = (a**2 + b**2 + c**2)**(.5)

data = np.concatenate((x[:, np.newaxis], 
                       y[:, np.newaxis], 
                       z[:, np.newaxis]), 
                      axis=1)

datamean = data.mean(axis=0)

uu, dd, vv = np.linalg.svd(data - datamean)

linepts = vv[0] * np.mgrid[-d/2:d/2:2j][:, np.newaxis]

linepts += datamean

show3d(data.T, lines=linepts.T)