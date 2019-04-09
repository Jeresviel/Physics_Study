import numpy as np

#基于numpy库编写的高斯消元法

def lu(matrix):
    "输入一个非奇异函数，利用递归方法对其进行LU分解，输出LU矩阵"
    if np.linalg.det(matrix): #判断矩阵奇异性
        if matrix.shape >= (2,2):
            a11,a12,a21,a22 = np.mat(matrix[0,0]),matrix[:1,1:],matrix[1:,:1],matrix[1:,1:] #对矩阵进行分块，使得左上角的矩阵为一阶矩阵
            l11,l21 = np.mat(1),(a21 / a11)
            u11,u12 = a11,a12 #分块后的LU矩阵，根据矩阵的运算法则对可以直接计算的矩阵计算
            l22,u22 = lu(a22 - l21 * u12) #对右下角的矩阵继续进行LU分解，实现递归
            l = np.hstack((np.vstack((l11,l21)),np.vstack((np.mat(np.zeros((1,l22.shape[1]))),l22))))
            u = np.vstack((np.hstack((u11,u12)),np.hstack((np.mat(np.zeros((u22.shape[0],1))),u22))))
        elif matrix.shape == (1,1):
            l,u = np.mat([1]),matrix #当矩阵为一阶时，根据定义直接返回LU矩阵
        else:
            print("矩阵输入错误！")
        return l,u
    else:
        print("该矩阵为奇异矩阵！")

def palud(matrix):
    "输入一个非奇异函数，利用递归方法对其进行PA = LU（部分选主元）分解，输出PLU矩阵"
    p = np.mat(np.eye(max(matrix.shape[0],matrix.shape[1])))
    def lud(matrix):
        if np.linalg.det(matrix):
            nonlocal p #记录行变换的P矩阵，由于使用递归，需要声明变量在整个函数内可全局调用，但在函数外视为局部变量
            if matrix.shape >= (2,2):
                if not np.argmax(matrix[:,0]): #如果最大值序号为0则不需要交换主元
                    matrix[[0,np.argmax(matrix[:,0])],:],p[[0,np.argmax(matrix[:,0])],:] = matrix[[np.argmax(matrix[:,0])],:],p[[np.argmax(matrix[:,0])],:]
                a11,a12,a21,a22 = np.mat(matrix[0,0]),matrix[:1,1:],matrix[1:,:1],matrix[1:,1:]
                l11,l21 = np.mat(1),(a21 / a11)
                u11,u12 = a11,a12
                l22,u22 = lud(a22 - l21 * u12)
                l = np.hstack((np.vstack((l11,l21)),np.vstack((np.mat(np.zeros((1,l22.shape[1]))),l22))))
                u = np.vstack((np.hstack((u11,u12)),np.hstack((np.mat(np.zeros((u22.shape[0],1))),u22))))
            elif matrix.shape == (1,1):
                l,u = np.mat([1]),matrix
            else:
                print("矩阵输入错误！")
            return l,u
        else:
            print("该矩阵为奇异矩阵！")
    l,u = lud(matrix)
    return l,u,p

def il(matrix):
    "对L矩阵进行逆运算"
    mat = matrix.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i > j:
                mat[i,j] = -mat[i,j]
    if mat * matrix != np.mat(np.eye(mat)):
        mat *= il(mat * matrix) #下三角矩阵相乘仍为下三角矩阵，递归后总能返回正确的逆矩阵，可以直接求得c矩阵
    else:
        return mat

def bsp(l,u,b):
    "高斯消元法的回代过程，即对b矩阵乘以U矩阵的逆"
    c = l.I * b
    x = np.mat(np.zeros((c.shape[0],1)))
    for i in range(x.shape[0] - 1,-1,-1):
        for j in range(x.shape[0] - 1,i,-1):
            c[i,0] -= u[i,j] * x[j,0]
        x[i,0] = c[i,0] / u[i,i]
    return x

def ge(A,b,p = True):
    "使用高斯消元法计算求解线性方程组，可选择是否使用部分选主元算法"
    if p: #如果使用部分选主元，则返回True
        l,u,p = palud(A)
        b = p * b
    else:
        l,u = lu(A)
    x = bsp(l,u,b)
    for i in range(x.shape[0]):
        print("x%d = %f" % (i+1,x[i,0]))

#基于高斯消元法的逆矩阵算法

def inv(matrix): #由于已经给出逆矩阵求解方法，之后需要求逆矩阵则直接调用，不再根据特殊情况手动求逆
    if np.linalg.det(matrix):
        E = np.mat(np.eye(matrix.shape[0])) #利用单位阵，按列求待逆矩阵为系数矩阵的解，最后得到逆矩阵
        for i in range(E.shape[1]):
            l,u,p = palud(matrix)
            E[:,i] = p * E[:,i]
            E[:,i] = bsp(l,u,E[:,i])
        return E
    else:
        print("矩阵奇异，无逆矩阵")

#基于numpy库编写的雅可比方法

def isddm(matrix):
    "对矩阵进行严格对角占优检验"
    mat = matrix.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i,j] = abs(mat[i,j]) 
    if ((mat.sum(axis = 1) - 2 * np.max(mat,1)).max() < 0) and (len(set(np.argmax(mat,1).T.tolist()[0])) == mat.shape[0]):
        return True,np.argmax(mat,1).T.tolist()[0]
    else:
        return False,np.argmax(mat,1).T.tolist()[0]

def ddm(matrix):
    "将矩阵化为严格主对角占优"
    mat = matrix.copy()
    re,maxlist = isddm(mat)
    if re:
        for i in range(len(maxlist)):
            if i != maxlist[i]:
                mat[[i,maxlist[i]],:] = mat[[maxlist[i],i],:]
    else:
        print("该矩阵不为严格主对角占优矩阵，可能导致迭代法不收敛")
        for i in range(len(maxlist)):
            if i != maxlist[i] and (mat[i,maxlist[i]] > mat[maxlist[i],maxlist[i]]): #如果异行最大值在同列，取更大的最大值
                mat[[i,maxlist[i]],:] = mat[[maxlist[i],i],:]
    return mat

def dlu(matrix):
    d = np.mat(np.zeros((matrix.shape[0],matrix.shape[1])))
    l = np.mat(np.zeros((matrix.shape[0],matrix.shape[1])))
    u = np.mat(np.zeros((matrix.shape[0],matrix.shape[1])))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
               d[i,j] = matrix[i,j]
            elif i > j:
               l[i,j] = matrix[i,j]
            elif i < j:
               u[i,j] = matrix[i,j]
    return d,l,u

def jac(A,b):
    d,l,u = dlu(ddm(A))
    x = np.mat(np.zeros((b.shape[0],1)))
    TOL = float(input("请输入精确度："))
    for i in range(A.shape[0]):
        x[i,0] = float(input("请输入初始向量第%d个分量：" % (i + 1)))
    s = [x]
    while True:
        s.append(d.I * (b - (l + u) * s[-1]))
        if ((s[-1] - s[-2]).max()) < TOL and abs((s[-1] - s[-2]).min()) < TOL:
            break
        del s[0]
    return s[-1]

#基于numpy库编写的高斯-赛德尔法

def gs(A,b):
    d,l,u = dlu(ddm(A))
    x1 = np.mat(np.zeros((b.shape[0],1)))
    TOL = float(input("请输入精确度："))
    for i in range(A.shape[0]):
        x1[i,0] = float(input("请输入初始向量第%d个分量：" % (i + 1)))
    s = [x1]
    d = d.I
    while True:
        x2 = np.mat(np.zeros((b.shape[0],1)))
        for i in range(x2.shape[0]): #由于使用最新估计值，不能矩阵数值计算，需要再增加循环数
            x2[i,0] = d[i,i] * (b[i,:] - u[i,:] * s[-1] - l[i,:] * x2)
        s.append(x2)
        if abs(((s[-1] - s[-2]).max())) < TOL and abs((s[-1] - s[-2]).min()) < TOL:
            break
        del s[0]
    return s[-1]

#基于numpy库编写的SOR（连续过松弛）方法

def sor(A,b):
    d,l,u = dlu(ddm(A))
    x = np.mat(np.zeros((b.shape[0],1)))
    TOL = float(input("请输入精确度："))
    omega = float(input("请输入松弛参数："))
    for i in range(A.shape[0]):
        x[i,0] = float(input("请输入初始向量第%d个分量：" % (i + 1)))
    s = [x]
    while True:
        s.append((omega * l + d).I * ((1 - omega) * d - omega * u) * s[-1] + omega * (d + omega * l).I * b)
        if ((s[-1] - s[-2]).max()) < TOL and abs((s[-1] - s[-2]).min()) < TOL:
            break
        del s[0]
    return s[-1]
