import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import sign
import os
import sys

def sort_points(filename, outfile):
    # parent = os.path.dirname(__file__)
    # filename = os.path.join(parent, 'data', 'airfoil.msh')
    x, y, z = np.loadtxt(filename, delimiter=' ', unpack=True)

    xsort = [x[0]]
    ysort = [y[0]]

    plt.figure()
    while(len(xsort) < len(x)):
        dx = x - xsort[-1]
        dy = y - ysort[-1]
        d = np.sqrt(dx**2 + dy**2)
        idx = np.argsort(d)
        dirx = 0
        diry = 0
        mod = 0
        if len(xsort) > 1:
            dirx = xsort[-2] - xsort[-1]
            diry = ysort[-2] - ysort[-1]
            mod = np.sqrt(dirx**2 + diry**2)
        dirx_all = xsort[-1] - x
        diry_all = ysort[-1] - y
        mod_all = np.sqrt(dirx_all**2 + diry_all**2)
        dot1 = (dirx*dirx_all + diry*diry_all)/(mod * mod_all)
        if (len(xsort) == 1):
            xsort.append(x[idx[1]])
            ysort.append(y[idx[1]])
        else:
            for id in idx:
                if (dot1[id] > 1e-14) and (not ((x[id] in xsort) and (y[id] in ysort))):
                    xsort.append(x[id])
                    ysort.append(y[id])
                    break
        plt.cla()
        plt.plot(x, y, '.')
        plt.plot(xsort, ysort)
        plt.pause(0.1)

    xsort.append(xsort[0])
    ysort.append(ysort[0])
    plt.show()

    # outfile = os.path.join(parent, 'data', 'airfoil.csv')
    np.savetxt(outfile , np.c_[xsort,ysort], fmt='%.3f')


if __name__ == '__main__':
    filename = sys.argv[-1]
    outfile = filename.split('.')[0] + '_sorted.csv'
    sort_points(filename, outfile)
    print('Done')
