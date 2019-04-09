from sympy import *

def mysolve(f,var,method = "bisection"):
    "用来解方程的函数，共有三个变量，f为待求解方程，var为待求解变量（仅支持一元方程），method为方法，目前提供二分法、牛顿法、弦截法、逆二次插值法。"
    if method == "bisection":#使用二分法
        TOL = float(input("请输入精度："))
        while True:
            a = float(input("请输入区间下限："))
            b = float(input("请输入区间上限："))
            if f.evalf(subs={var:a}) * f.evalf(subs={var:b}) > 0:
                print('区间范围内错误')
                continue
        while (b-a)/2 > TOL:
            c = (a+b)/2
            if not f.evalf(subs={var:c}):
                break
            elif f.evalf(subs={var:a}) * f.evalf(subs={var:c}) < 0:
                b = c
            elif f.evalf(subs={var:b}) * f.evalf(subs={var:c}) < 0:
                a = c
        print("近似根为 %f" % ((a + b)/2))
    elif method == "Newton":#使用牛顿法
        TOL = float(input("请输入精确度："))
        x = [float(input("请输入初始估计值："))]
        while True:
            x.append(x[-1] - f.evalf(subs={var:x[-1]}) / f.diff().evalf(subs={var:x[-1]}))
            if abs(x[-2] - x[-1]) < TOL:
                break
            del x[0]
        print("方程的根为 %f" % (x[-1]))
    elif method == "secant":#使用弦截法
        TOL = float(input("请输入精度："))
        x = [float(input("请输入初始估计值1：")),float(input("请输入初始估计值2："))]
        while abs(x[-2] - x[-1]) > TOL:
           x.append(x[-1] - (f.evalf(subs={var:x[-1]}) * (x[-1]-x[-2]))/(f.evalf(subs={var:x[-1]})-f.evalf(subs={var:x[-2]})))
           del x[0]
        print("近似根为 %f" % (x[-1]))
    elif method == "IQI":#使用逆二次插值法
        TOL = float(input("请输入精确度："))
        x = [float(input("请输入估计值1：")),float(input("请输入估计值2：")),float(input("请输入估计值3："))]
        absf = lambda n : abs(f.evalf(subs={var:n}))
        ref = lambda m : f.evalf(subs={var:m})
        rep = input("是否用最新估计值替换最近三个估计值中后向误差最大项？否则将替代最旧估计值。\ny/n:")
        while True:
            q = f.evalf(subs={var:x[-3]}) / f.evalf(subs={var:x[-2]})
            r = f.evalf(subs={var:x[-1]}) / f.evalf(subs={var:x[-2]})
            s = f.evalf(subs={var:x[-1]}) / f.evalf(subs={var:x[-3]})
            x.append(x[-1] - (r * (r - q) * (x[-1] - x[-2]) + (1 - r) * s * (x[-1] - x[-3])) / ((q - 1) * (r - 1) * (s - 1)))
            if len(x) != len(set(x)):
                print("出现重复估计值 ",end="")
                break
            elif rep == "y":
                del x[list(map(absf,x[:3])).index(max(list(map(absf,x[:3]))))]
            else:
                del x[0]
            if (abs(x[-1] - x[-2]) < TOL) and (abs(x[-2] - x[-3]) < TOL) and (abs(x[-3] - x[-1]) < TOL):
                print("近似根为 ",end="")
                break
        print(x[list(map(ref,x)).index(min(list(map(ref,x))))])
    else:
        print("不支持此方法！")
