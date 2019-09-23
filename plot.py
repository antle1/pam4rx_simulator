import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
plt.figure(1) # the first figure
plt.plot([1,2,3])
plt.figure(2) # a second figure
plt.plot([4,5,6])
plt.show()