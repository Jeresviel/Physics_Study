import numpy as np
import matplotlib.pyplot as plt

def df(f,area,step):
    df = np.array([])
    area = np.arange(area[0],area[1]+step,step)
    for i in area:
        df = np.append(df,[(f(i-step)-8*f(i-step/2)+8*f(i+step/2)-f(i+step))/(6*step)],axis=0)
    return df

def partial_x(f,y,area,step):
    dx = np.array([])
    area = np.arange(area[0],area[1]+step,step)
    for i in area:
        dx = np.append(dx,[(f(i-step,y)-8*f(i-step/2,y)+8*f(i+step/2,y)-f(i+step,y))/(6*step)],axis=0)
    return dx

def partial_y(f,x,area,step):
    dy = np.array([])
    area = np.arange(area[0],area[1]+step,step)
    for i in area:
        dy = np.append(dy,[(f(x,i-step)-8*f(x,i-step/2)+8*f(x,i+step/2)-f(x,i+step))/(6*step)],axis=0)
    return dy

def grad_2D(f,area,step):
    areax,areay = np.arange(area[0,0],area[0,1]+step[0],step[0]),np.arange(area[1,0],area[1,1]+step[1],step[1])
    gradient = np.empty((areax.shape[0],areay.shape[0],2))
    for i in range(areay.shape[0]):
        gradient[:,i,0] = partial_x(f,areay[i],area[0],step[0])
    for i in range(areax.shape[0]):
        gradient[i,:,1] = partial_y(f,areax[i],area[1],step[1])
    return gradient

def div_2D(f,area,step):
    gradient = grad_2D(f,area,step)
    return gradient.sum(axis = 2)

def f(x):
    return np.log(x)

def g(x,y):
    return np.exp(x) + np.log(y+1.1)

y1 = df(f,(0.1,10),0.01)
x1 = np.arange(0.1,10.01,0.01)
y2 = div_2D(g,np.array([[-1,1],[-1,1]]),(0.1,0.1))
plt.figure()
plt.plot(x1,y1)
plt.show()