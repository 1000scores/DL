import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib as mpl
#import matplotlib_terminal
#mpl.use ( 'TkAgg')
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()
#time.sleep(100)