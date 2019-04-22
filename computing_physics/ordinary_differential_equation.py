import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def RK2(df,f0,area,h):#龙格-库塔2阶方法
    t = np.arange(area[0],area[1] + h,h)
    h = h * np.ones(f0[0].shape)
    f = np.array([f0])
    for i in t[:-1]:
        f = np.append(f,[f[-1] + h * df(i + h/2,f[-1] + h/2 * df(i,f[-1]))],axis = 0)
    return t,f

def RK4(df,f0,area,h):#龙格-库塔4阶方法方法
    t = np.arange(area[0],area[1] + h,h)
    h = h * np.ones(f0[0].shape)
    f = np.array([f0])
    def slope(df,i,f,h):
        s1 = df(i,f[-1])
        s2 = df(i + h/2,f[-1] + h/2 * s1)
        s3 = df(i + h/2,f[-1] + h/2 * s2)
        s4 = df(i + h,f[-1] + h * s3,)
        return s1 + 2 * s2 + 2 * s3 + s4
    for i in t[:-1]:
        f = np.append(f,[f[-1] + h/6 * slope(df,i,f,h)],axis=0)
    return t,f

def leap_frog(df,f0,area,h):
    t = np.arange(area[0],area[1]+h,h/2)
    h = h * np.ones(f0[0].shape)
    f = np.append([f0],[f0 + h * df(t[0],f0)],axis = 0)
    for i in t[1:-1:2]:
        f = np.append(f,[f[-2] + h * df(i+h/2,f[-1])],axis = 0)
        f = np.append(f,[f[-2] + h * df(i+h,f[-1])],axis = 0)
    return t,f
#输出的f为一个三维数组，f[x,y,z]，x表示t轴，y表示不同的微分方程的解，z表示变量的维度
def simple_pendulum(t,p):
    m,g,l = 1,9.8,1
    dtheta = lambda t,p: p[1]
    def domega(t,p):
        nonlocal g,l
        return -g/l * np.sin(p[0])
    return np.array([dtheta(t,p),domega(t,p)])

def Lorenz(t,p):
    s,r,b = 10,28,8/3
    dx = lambda t,p : s * (p[1] - p[0])
    dy = lambda t,p : r * p[0] - p[1] - p[0] * p[2]
    dz = lambda t,p : p[0] * p[1] - b * p[2]
    return np.array([dx(t,p),dy(t,p),dz(t,p)])
#对应的微分方程定义，输入p为矢量，输出也是一个矢量（或矩阵），p中变量顺序和输出是对应的