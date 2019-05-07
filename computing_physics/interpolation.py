import numpy as np
import matplotlib.pyplot as plt

def L(x,data):
    sum = 0
    for i in range(data.shape[1]):
        l = 1
        for j in range(data.shape[1]):
            if i == j:
                continue
            else:
                l *= (x - data[0,j])/(data[0,i] - data[0,j])
        sum += data[1,i]*l
    return sum

def C(f,area,n):
    point = np.array([])
    for i in range(n):
        point = np.append(point,area.mean() + (area[1]-area[0])/2*np.cos((2*i+1)*np.pi/(2*n)))
    value = np.array([f(i) for i in point])
    return np.append([point],[value],axis = 0)

def f(x):
    return 1/(1+12*x**2)

data = np.array([[-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]])
x = np.linspace(-1,1,15)
y = np.array([f(i) for i in x])
data1 = np.array([x,y])
data2 = C(f,np.array([-1,1]),15)
x = np.linspace(-1.05,1.05,100)
y1 = np.array([L(i,data1) for i in x])
y2 = np.array([L(i,data2) for i in x])
x3 = np.linspace(-5.5,5.5,100)
y3 = np.array([L(i,data) for i in x3])
y = np.array([f(i) for i in x])
plt.figure()
plt.plot(x,y1,c='gray',label='$Lagrange\\ interpolation$')
plt.plot(x,y2,'--',c='gray',label='$Chebyshev\\ interpolation$')
plt.plot(x,y,'-.',c='gray',label='$Value\\ of\\ function\\ y=\\frac{1}{1+12x^2}$')
plt.legend()
plt.figure()
plt.title("$Runge's\\ phenomenon$")
plt.plot(x3,y3,c='gray')
plt.scatter(data[0][3:-3],data[1][3:-3],c='gray',marker='.')
plt.show()