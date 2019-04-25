import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def poisson(f,boundary_value,area):
    m,n = area[0].shape[0],area[1].shape[0]
    h,k = area[0][1]-area[0][0],area[1][1]-area[1][0]
    A,b = np.mat(np.zeros((m*n,m*n))),np.mat(np.zeros((m*n,1)))
    for i in range(1,m-1):
        for j in range(1,n-1):
            A[i+j*m,i-1+j*m] = 1/(h**2)
            A[i+j*m,i+1+j*m] = 1/(h**2)
            A[i+j*m,i+j*m] = -2/(h**2) -2/(k**2)
            A[i+j*m,i+(j-1)*m] = 1/(k**2)
            A[i+j*m,i+(j+1)*m] = 1/(k**2)
            b[i+j*m,0] = f(area[0][i],area[1][j])
    for i in range(0,m):
        j = 0
        A[i+j*m,i+j*m],b[i+j*m,0] = 1,boundary_value(area[0][i],area[1][j])
        j = n-1
        A[i+j*m,i+j*m],b[i+j*m,0] = 1,boundary_value(area[0][i],area[1][j])
    for j in range(1,n-1):
        i = 0
        A[i+j*m,i+j*m],b[i+j*m,0] = 1,boundary_value(area[0][i],area[1][j])
        i = m-1
        A[i+j*m,i+j*m],b[i+j*m,0] = 1,boundary_value(area[0][i],area[1][j])
    v = A.I * b
    v= np.array(v).reshape((m,n)).T
    return v

def boundary_value_1(x,y):
    g1 = lambda x,y: np.log(x**2+1)
    g2 = lambda x,y: np.log(x**2+4)
    g3 = lambda x,y: 2*np.log(y)
    g4 = lambda x,y: np.log(y**2+1)
    if y == 1:
        return g1(x,y)
    elif y == 2:
        return g2(x,y)
    elif x == 0:
        return g3(x,y)
    elif x == 1:
        return g4(x,y)

def boundary_value_2(x,y):
    g1 = lambda x,y: np.sin(np.pi*x)
    g2 = lambda x,y: np.sin(np.pi*x)
    g3 = lambda x,y: -np.sin(np.pi*y)
    g4 = lambda x,y: np.sin(np.pi*y)
    if y == 0:
        return g1(x,y)
    elif y == 1:
        return g2(x,y)
    elif x == 0:
        return g3(x,y)
    elif x == 1:
        return g4(x,y)

f = lambda x,y : 0
x1,y1 = np.linspace(0,1,30),np.linspace(1,2,30)
x2,y2 = np.linspace(0,1,30),np.linspace(0,1,30)
X1,Y1 = np.meshgrid(x1,y1)
X2,Y2 = np.meshgrid(x2,y2)
z1 = poisson(f,boundary_value_1,(x1,y1))
z2 = poisson(f,boundary_value_2,(x2,y2))
fig1 = plt.figure()
ax1 = Axes3D(fig1)
ax1.plot_surface(X1,Y1,z1)
plt.xlabel("x")
plt.ylabel("y")
fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.plot_surface(X2,Y2,z2)
plt.show()