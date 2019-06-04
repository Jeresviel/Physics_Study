import numpy as np
import sympy as sp
from numba import autojit
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import time

def Simpson(f,area):
    area = np.array(area)
    return (area[1]-area[0])/6 * (f(area[0])+4*f(area.mean())+f(area[1]))

def open_interval_Newton_Cortes(f,area):
    h = (area[1]-area[0])/4
    return 4*h/3 * (2*f(area[0]+h)-f(area[0]+2*h)+2*f(area[0]+3*h))

def composite_Simpson(f,area,h):
    x = np.arange(area[0],area[1]+h,h)
    y = np.array([f(i) for i in x])
    return h/3 * (y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-2:2].sum())

def Simpson_2D(f,area):
    area = np.array(area)
    return (area[0,1]-area[0,0])*(area[1,1]-area[1,0])/36 * (f(area[0,0],area[1,0])+f(area[0,0],area[1,1])+f(area[0,1],area[1,0])+f(area[0,1],area[1,1])+4*(f(area[0,0],area[1].mean())+f(area[0].mean(),area[1,0])+f(area[0,1],area[1].mean())+f(area[0].mean(),area[1,1]))+16*(f(area[0].mean(),area[1].mean())))

def composite_Simpson_2D(f,area,h):
    x = np.arange(area[0,0],area[0,1]+h[0],h[0])
    y = np.arange(area[1,0],area[1,1]+h[1],h[1])
    area = np.empty((x.shape[0],y.shape[0],2))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            area[i,j] = (x[i],y[j])
    integral = 0
    for i in range(area.shape[0]-1):
        for j in range(area.shape[1]-1):
            integral += Simpson_2D(f,np.array([area[i,j],area[i+1,j+1]]).T)
    return integral

def Romberg(f,area,tolerance = 1E-6):
    h = area[1]-area[0]
    R = [np.array([h/2*(f(area[0])+f(area[1]))])]
    while True:
        h = h/2
        Ri = np.array([R[-1][0]/2 + sum([h*f(area[0]+(2*i+1)*h) for i in range(2**(len(R)-1))])])
        for i in range(1,len(R)+1):
            Ri = np.append(Ri,(4**i*Ri[-1] - R[-1][i-1])/(4**i-1))
        R.append(Ri)
        if abs(R[-1][-1] - R[-1][-2]) < tolerance:
            break
    return R[-1][-1]

def self_adaptive_Simpson(f,area,tolerance = 1E-6):
    origin = area[1] - area[0]
    area,value = np.array([area]),np.array([Simpson(f,area)])
    integral = 0
    while True:
        area1,area2,area3 = area[-1,-2:],(area[-1,-2],area[-1,-2:].mean()),(area[-1,-2:].mean(),area[-1,-1])
        S1,S2,S3 = value[-1],Simpson(f,area2),Simpson(f,area3)
        if abs(S1 - (S2 + S3)) < 15*(area1[1]-area1[0])/origin*tolerance:
            integral += S2 + S3
            area,value = np.delete(area,-1,axis=0),np.delete(value,-1,axis=0)
        else:
            area,value = np.delete(area,-1,axis=0),np.delete(value,-1,axis=0)
            area,value = np.concatenate((area,[area2],[area3]),axis=0),np.concatenate((value,[S2],[S3]),axis=0)
        if not area.shape[0]:
            break
    return integral

def self_adaptive_Simpson_origin(f,area,tolerance = 1E-6):
    origin = area[1] - area[0]
    area = np.array([area])
    integral = 0
    while True:
        area1,area2,area3 = area[-1,-2:],(area[-1,-2],area[-1,-2:].mean()),(area[-1,-2:].mean(),area[-1,-1])
        S1,S2,S3 = Simpson(f,area1),Simpson(f,area2),Simpson(f,area3)
        if abs(S1 - (S2 + S3)) < 15*(area1[1]-area1[0])/origin*tolerance:
            integral += S2 + S3
            area = np.delete(area,-1,axis=0)
        else:
            area = np.delete(area,-1,axis=0)
            area = np.concatenate((area,[area2],[area3]),axis=0)
        if not area.shape[0]:
            break
    return integral

def self_adaptive_open_interval(f,area,tolerance = 1E-6):
    origin = area[1] - area[0]
    area,value = np.array([area]),np.array([open_interval_Newton_Cortes(f,area)])
    integral = 0
    while True:
        area1,area2,area3 = area[-1,-2:],(area[-1,-2],area[-1,-2:].mean()),(area[-1,-2:].mean(),area[-1,-1])
        S1,S2,S3 = value[-1],open_interval_Newton_Cortes(f,area2),open_interval_Newton_Cortes(f,area3)
        if abs(S1 - (S2 + S3)) < 15*(area1[1]-area1[0])/origin*tolerance:
            integral += S2 + S3
            area,value = np.delete(area,-1,axis=0),np.delete(value,-1,axis=0)
        else:
            area,value = np.delete(area,-1,axis=0),np.delete(value,-1,axis=0)
            area,value = np.concatenate((area,[area2],[area3]),axis=0),np.concatenate((value,[S2],[S3]),axis=0)
        if not area.shape[0]:
            break
    return integral

def line_integral(Ex,Ey,area,h):
    X = np.arange(area[0,0],area[0,1]+h[0],h[0])
    Y = np.arange(area[1,0],area[1,1]+h[1],h[1])
    phi = np.empty((X.shape[0],Y.shape[0],3))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            f = lambda t : X[i]/t**2 * Ex(X[i]/t,Y[j]/t) + Y[j]/t**2 * Ey(X[i]/t,Y[j]/t)
            phi[i,j,:2] = (X[i],Y[j])
            phi[i,j,2] = self_adaptive_open_interval(f,(0,1))
    return phi

def Gauss(f,area,n=8):
    root = np.loadtxt("root.txt")
    coefficient = np.loadtxt("coefficient.txt")
    if n < root.shape[0] :
        integral = np.array([(area[1]-area[0])/2*coefficient[n,i]*f(((area[1]-area[0])*root[n,i]+sum(area))/2) for i in range(n)])
        return integral.sum()
    else:
        print("Error: n is out of range of 0 to %d" %(root.shape[0]-1))

def composite_Gauss(f,area,h,n=10):
    area = np.arange(area[0],area[1]+h,h)
    integral = 0
    for i in area[:-1]:
        integral += Gauss(f,(i,i+h),n)
    return integral

def self_adaptive_Gauss(f,area,tolerance = 1E-6):
    origin = area[1] - area[0]
    area,value = np.array([area]),np.array([Gauss(f,area)])
    integral = 0
    while True:
        area1,area2,area3 = area[-1,-2:],(area[-1,-2],area[-1,-2:].mean()),(area[-1,-2:].mean(),area[-1,-1])
        S1,S2,S3 = value[-1],Gauss(f,area2),Gauss(f,area3)
        if abs(S1 - (S2 + S3)) < 15*(area1[1]-area1[0])/origin*tolerance:
            integral += S2 + S3
            area,value = np.delete(area,-1,axis=0),np.delete(value,-1,axis=0)
        else:
            area,value = np.delete(area,-1,axis=0),np.delete(value,-1,axis=0)
            area,value = np.concatenate((area,[area2],[area3]),axis=0),np.concatenate((value,[S2],[S3]),axis=0)
        if not area.shape[0]:
            break
    return integral

def Bessel_functions_Romberg(x):
    f = lambda theta : np.cos(theta - x*np.sin(theta))
    return 1/np.pi * Romberg(f,(0,np.pi))

def Bessel_functions_self_adaptive(x):
    f = lambda theta : np.cos(theta - x*np.sin(theta))
    return 1/np.pi * self_adaptive_Simpson(f,(0,np.pi))

def Bessel_functions_Gauss(x):
    f = lambda theta : np.cos(theta - x*np.sin(theta))
    return 1/np.pi * self_adaptive_Gauss(f,(0,np.pi))

def Fraunhofer_diffraction_Romberg(x):
    return np.array([(2*Bessel_functions_Romberg(i)/i)**2 for i in x])

def Fraunhofer_diffraction_self_adaptive(x):
    return np.array([Bessel_functions_self_adaptive(i)/i for i in x])

def Fraunhofer_diffraction_Gauss(x):
    return np.array([Bessel_functions_self_adaptive(i)/i for i in x])

def Fraunhofer_diffraction_3D(I,Z):
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            I[i,j,0],I[i,j,1] = Z[i],Z[j]
            a = np.linalg.norm(I[i,j,0:2])
            I[i,j,2] = ((2*Bessel_functions_Romberg(a)/a)**2)
    return I

#圆孔的夫琅禾费衍射图样
delta = np.linspace(-15,15,300)
X,Y = np.meshgrid(delta,delta)
I_3D = np.empty((delta.shape[0],delta.shape[0],3))
t1 = time()
I1 = Fraunhofer_diffraction_Romberg(delta)
t2 = time()
I2 = Fraunhofer_diffraction_self_adaptive(delta)
t3 = time()
I3 = Fraunhofer_diffraction_Gauss(delta)
t4 = time()
print("龙贝格积分：%f秒,自适应积分：%f秒,高斯积分：%f秒" %(t2-t1,t3-t2,t4-t3))
I4 = Fraunhofer_diffraction_3D(I_3D,delta)
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,2,1)
ax1.plot(delta,I1)
ax2 = fig1.add_subplot(2,2,2)
ax2.plot(delta[80:221],I1[80:221])
ax3 = fig1.add_subplot(2,2,3)
ax3.imshow(I4[:,:,2]**(1/2),cmap="gray")
ax3.axis("off")
ax4 = fig1.add_subplot(2,2,4,projection="3d")
ax4.plot_surface(X,Y,I4[:,:,2])
plt.show()
#求解积分限较大时的积分
f = lambda x : np.log(10)*(10**(3*x))
t1 = time()
i1 = Romberg(f,(0,8),1)
t2 = time()
#i2 = self_adaptive_Simpson(f,(0,8),1)
t3 = time()
i3 = self_adaptive_Gauss(f,(0,8),1)
t4 = time()
print("龙贝格积分：%f秒,自适应积分：%f秒,高斯积分：%f秒" %(t2-t1,t3-t2,t4-t3))