import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def RK4(df,f0,range,h):#这实际上就是Matlab中的ODE45，但是需要自己设定步长
    t = np.arange(range[0],range[1] + h,h)
    h = h * np.ones(f0.shape)
    f = np.array([f0])
    for i in t[:-1]:
        s1 = df(i,f[-1])
        s2 = df(i + h/2,f[-1] + h/2 * s1)
        s3 = df(i + h/2,f[-1] + h/2 * s2)
        s4 = df(i + h,f[-1] + h * s3)
        f = np.append(f,[f[-1] + h/6 * (s1 + 2 * s2 + 2 * s3 + s4)],axis = 0)
        print(f)
    return t,f