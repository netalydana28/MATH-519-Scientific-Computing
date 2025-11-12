import numpy as np
import matplotlib.pyplot as plt

numerical_data = np.loadtxt("sequential.txt")
analytical_data = np.loadtxt("parallel.txt")

x_num, y_num, u_num = numerical_data[:,0], numerical_data[:,1], numerical_data[:,2]
x_anal, y_anal, u_anal = analytical_data[:,0], analytical_data[:,1], analytical_data[:,2]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x_num, y_num, c=u_num, cmap="viridis")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sequential solution")

plt.subplot(1, 2, 2)
plt.scatter(x_anal, y_anal, c=u_anal, cmap="viridis")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Parallel solution")

plt.tight_layout()
plt.show()
