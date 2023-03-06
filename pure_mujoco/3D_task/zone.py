from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

coeff=2
fig = plt.figure(figsize=(25,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("auto")

# # draw cube
# r = [-1, 1]
# for s, e in combinations(np.array(list(product(r, r, r))), 2):
#     if np.sum(np.abs(s-e)) == r[1]-r[0]:
#         ax.plot3D(*zip(s, e), color="b")

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]#params for the numbers of row lines and column lines 20 10
x = np.cos(u)*np.sin(v)*coeff
y = np.sin(u)*np.sin(v)*coeff
z = np.cos(v)*coeff
ax.plot_wireframe(x, y, z, color="m",linewidth=0.5)

# draw a point
ax.scatter([0], [0], [0], color="g", s=100)
ax.scatter([2],[0],[0],color="b", s=100)
ax.scatter([1.414],[0],[1.414],color="y", s=100)
ax.scatter([1.414],[1.414],[0],color="k", s=100)
ax.scatter([1.99],[0],[0.15],color="r", s=100)
ax.scatter([1.99],[0],[-0.15],color="r", s=100)
ax.scatter([1.99],[-0.15],[0],color="r", s=100)
ax.scatter([1.99],[0.15],[0],color="r", s=100)
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# # a random array of 3D coordinates in [-1,1]
# bvecs= np.random.randn(20,3)

# # tails of the arrows
# tails= np.zeros(len(bvecs))

# # heads of the arrows with adjusted arrow head length
# ax.quiver(tails,tails,tails,bvecs[:,0], bvecs[:,1], bvecs[:,2],
#           length=1.0, normalize=True, color='b', arrow_length_ratio=0.15)

plt.show()