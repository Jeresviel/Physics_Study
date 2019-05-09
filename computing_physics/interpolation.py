import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def Lagrange(x,y):
    polynomial,var = 0,sp.symbols("x")
    for i in range(x.shape[0]):
        L = 1
        for j in range(x.shape[0]):
            if i == j:
                continue
            else:
                L *= (var - x[j])/(x[i] -x[j])
        polynomial +=y[i]*L
    return sp.expand(polynomial),var

def Newton(x,y):
    f = []
    f.append(y)
    for i in range(1,y.shape[0]):
        fx = np.array([])
        for j in range(f[-1].shape[0]-1):
            fx = np.append(fx,(f[-1][j+1]-f[-1][j])/(x[i+j]-x[j]))
        f.append(fx)
    f = np.array([f[i][0] for i in range(len(f))])
    polynomial,var = 0,sp.symbols("x")
    x = var - x
    for i in range(f.shape[0]):
        polynomial += f[i]*x[0:i].prod()
    return sp.expand(polynomial),var

def Chebyshev(f,area,n):
    point = np.array([])
    for i in range(n):
        point = np.append(point,area.mean() + (area[1]-area[0])/2*np.cos((2*i+1)*np.pi/(2*n)))
    value = np.array([f(i) for i in point])
    return np.append([point],[value],axis = 0)

f = lambda x : 1/(1+12*x**2)
data1 = np.array([[-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]])
x = np.linspace(-1,1,15)
y = np.array([f(i) for i in x])
data2 = Chebyshev(f,np.array([-1,1]),15)
y1,x1 = Newton(x,y)
y2,x2 = Newton(data2[0],data2[1])
y3,x3 = Newton(data1[0],data1[1])
X,X2 = np.linspace(-1.05,1.05,100),np.linspace(-5.05,5.05,200)
Y = np.array([f(i) for i in X])
Y1 = np.array([y1.evalf(subs={x1:i}) for i in X])
Y2 = np.array([y2.evalf(subs={x2:i}) for i in X])
Y3 = np.array([y3.evalf(subs={x3:i}) for i in X2])
plt.figure()
plt.plot(X,Y1,c='gray',label='$Lagrange\\ interpolation$')
plt.plot(X,Y2,'--',c='gray',label='$Chebyshev\\ interpolation$')
plt.plot(X,Y,'-.',c='gray',label='$Value\\ of\\ function\\ y=\\frac{1}{1+12x^2}$')
plt.legend()
plt.figure()
plt.title("$Runge's\\ phenomenon$")
plt.plot(X2,Y3,c='gray')
plt.scatter(data1[0][3:-3],data1[1][3:-3],c='gray',marker='.')
plt.show()