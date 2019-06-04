from random import random
import numpy as np

def buffon(a,b,n):
    location,phi = [a/2 * random() for i in range(n)],[np.pi * random() for i in range(n)]
    result = sum([i <= 0 and 1 or 0 for i in map(lambda x,y : x - b/2 * np.sin(y),location,phi)])
    return 2*n/result*b/a

a = float(input("请输入横线距离："))
b = float(input("请输入针长度："))
n = int(input("请输入投针次数："))
pi = buffon(a,b,n)
print("投针实验的结果为%f" %pi)