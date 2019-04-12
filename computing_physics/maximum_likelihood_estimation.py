from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt


x = [58.0,125.0,131.0,146.0,157.0,158.0,160.0,165.0,166.0,186.0,198.0,201.0,202.0,210.0,218.0]
y = [173.0,334.0,311.0,344.0,317.0,416.0,337.0,393.0,400.0,423.0,510.0,442.0,504.0,479.0,533.0]
sigma = [15.0,26.0,16.0,22.0,52.0,16.0,31.0,14.0,34.0,42.0,30.0,25.0,14.0,27.0,16.0]
def mle(x,y,sigma):
    def chi_square(ai):
        sum = 0
        nonlocal x,y,sigma
        for i in range(0,len(x)):
            sum += (ai[0] * x[i] + ai[1] - y[i]) ** 2 / (2 * sigma[i] ** 2)
        return sum
    return fmin(chi_square,np.array([1,1]))

def chi_square(a1,a2):
        sum = 0
        for i in range(0,len(x)):
            sum += (a1 * x[i] + a2 - y[i]) ** 2 / (2 * sigma[i] ** 2)
        return sum
result = mle(x,y,sigma)
plt.figure(1)
plt.subplot(1,2,1)
plt.title("最大似然估计")
x0 = np.arange(x[0],x[-1],0.05)
y0 = result[0] * x0 + result[1]
plt.errorbar(x,y,sigma,fmt = ".", elinewidth = 1, capsize = 2)
plt.xlabel('$x$')
plt.ylabel('$y$', rotation = 360)
plt.plot(x0, y0 , label = '$y=%.3fx+%.3f$' %(result[0],result[1]))
plt.legend()
plt.subplot(1,2,2)
plt.title("参数误差")
x1,y1 = np.linspace(2,2.5,500),np.linspace(5,70,500)
X,Y = np.meshgrid(x1,y1)
plt.contour(X,Y,chi_square(X,Y)-chi_square(result[0],result[1])-1,0)
plt.plot(result[0],result[1],".")
plt.text(result[0] - 0.1, result[1] - 2.5, "$(%.3f,%.3f)$" %(result[0],result[1]))
plt.xlabel('$a_1$')
plt.ylabel('$a_2$', rotation = 360)
plt.show()