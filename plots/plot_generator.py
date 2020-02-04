import numpy as np
import matplotlib.pyplot as plt
import os

root = os.path.normpath('C:/Users/John/Dropbox/UIUC/Research/')
my_data = np.genfromtxt(root + '/IDD_SlidingTileResults.csv', delimiter=',').T

labels = ["A*", "RA*", "NBS", "DVCBS", "GBFHS", "DIBBS", "IDD", "IDD-C"]

expansions = my_data[:, 0::3]
timing = my_data[:, 1::3]
memory = my_data[:, 2::3]

plt.title("Sliding Tile - Node Expansions")
plt.boxplot(memory, sym="", labels=labels)
plt.show()