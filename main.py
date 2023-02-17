import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from box import Box

box = Box([1,2,3],[1,1,1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
box.plot(ax,a=0.3)
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
