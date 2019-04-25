import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def RK2(df,f0,area,h):#龙格-库塔2阶方法
    t = np.arange(area[0],area[1] + h,h)
    f = np.array([f0])
    for i in t[:-1]:
        f = np.append(f,[f[-1] + h * df(i + h/2,f[-1] + h/2 * df(i,f[-1]))],axis = 0)
    return t,f

def RK4(df,f0,area,h):#龙格-库塔4阶方法方法
    t = np.arange(area[0],area[1] + h,h)
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

def Lorentz(t,p):
    m,q,E,B = 1,1,np.array([0,0,1]),np.array([0,0,1])
    dr = lambda t,p : p[1]
    def dv(t,p):
        nonlocal m,q,E,B
        return q/m * (E + np.cross(p[1],B))
    return np.array([dr(t,p),dv(t,p)])

def three_body_problem(t,p):
    m1,m2,m3,G = 0*1.898E27,5.972E24,1.989E30,6.67408E-11
    dr1 = lambda t,p : p[1]
    def dv1(t,p):
        nonlocal m1,m2,m3,G
        return G*m2/(np.linalg.norm(p[2]-p[0])**3)*(p[2]-p[0])+G*m3/(np.linalg.norm(p[4]-p[0])**3)*(p[4]-p[0])
    dr2 = lambda t,p : p[3]
    def dv2(t,p):
        nonlocal m1,m2,m3,G
        return G*m3/(np.linalg.norm(p[4]-p[2])**3)*(p[4]-p[2])+G*m1/(np.linalg.norm(p[0]-p[2])**3)*(p[0]-p[2])
    dr3 = lambda t,p : p[5]
    def dv3(t,p):
        nonlocal m1,m2,m3,G
        return G*m1/(np.linalg.norm(p[0]-p[4])**3)*(p[0]-p[4])+G*m2/(np.linalg.norm(p[2]-p[4])**3)*(p[2]-p[4])
    return np.array([dr1(t,p),dv1(t,p),dr2(t,p),dv2(t,p),dr3(t,p),dv3(t,p)])
#对应的微分方程定义，输入p为矢量，输出也是一个矢量（或矩阵），p中变量顺序和输出是对应的
t1,f1 = RK4(three_body_problem,np.array([[7.78E11,0,0],[0,1.307E4,0],[1.4959787E11,0,0],[0,2.98E4,0],[0,0,0],[0,0,0]]),[0,1E9],1E6)
t2,f2 = RK4(Lorentz,np.array([[0,0,0],[0,1,0]]),(0,20),0.1)
plt.figure()
plt.plot(f1[:,0,0],f1[:,0,1],label="$Jupiter$")
plt.plot(f1[:,2,0],f1[:,2,1],label="$Earth$")
plt.plot(f1[:,4,0],f1[:,4,1],label="$the Sun$")
plt.legend()
fig = plt.figure()
ax = fig.gca(projection = '3d')
x = f2[:,0,0]
y = f2[:,0,1]
z = f2[:,0,2]
plt.plot(x,y,z,label="Orbit of A Charged Particle")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()