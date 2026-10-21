from pathlib import Path
import numpy as np,json,math
records=[]
def check(group,name,a,b,tol=2e-7):
    error=float(np.max(np.abs(np.asarray(a)-b)))/max(1.,float(np.max(np.abs(b))))
    records.append(dict(group=group,name=name,normalized_error=error,tolerance=tol,passed=error<=tol))
g,w=np.polynomial.legendre.leggauss(180)
def integral(f,a,b):return (b-a)/2*np.dot(w,f((b-a)*g/2+(a+b)/2))
def phi(t,z):
    if z<1:
        v=math.sqrt(1-z*z);c=math.cos(v*t);s=math.sin(v*t)/v
    elif z==1:c=1;s=t
    else:
        v=math.sqrt(z*z-1);c=math.cosh(v*t);s=math.sinh(v*t)/v
    return math.exp(-z*t)*np.array([[c+z*s,s],[-s,c-z*s]])
def rk(x,t,z,e=0):
    # Independent integration of the two state equations, not the closed solution.
    n=1800;h=t/n;x=np.array(x,dtype=float)
    f=lambda a:np.array([a[1],e-a[0]-2*z*a[1]])
    for _ in range(n):
        k1=f(x);k2=f(x+h*k1/2);k3=f(x+h*k2/2);k4=f(x+h*k3);x+=h*(k1+2*k2+2*k3+k4)/6
    return x
for z in [.05,.3,.6,.99,1,1.01,1.25,3]:
    for x0 in [[1,0],[.5,-1],[0,2]]:
        for t in [.2,2,6]:check('独立微分方程','三阻尼响应',phi(t,z)@x0,rk(x0,t,z))
    check('状态矩阵','半群',phi(2,z)@phi(.7,z),phi(2.7,z))
    check('状态矩阵','面积压缩',np.linalg.det(phi(2,z)),math.exp(-4*z))
    x0=np.array([1.,-.4]);t=4;x=phi(t,z)@x0
    heat=integral(lambda s:np.array([2*z*(phi(q,z)@x0)[1]**2 for q in s]),0,t)
    check('独立积分','RLC能量',heat,.5*(np.dot(x0,x0)-np.dot(x,x)))
for E,V,R,C in [(12,-4,1000,1e-4),(3,5,2,.4),(-2,1,5,.2)]:
    tau=R*C;T=9*tau;I=lambda t:(E-V)/R*np.exp(-t/tau)
    check('独立积分','RC有限时热',integral(lambda t:R*I(t)**2,0,T),C*(E-V)**2/2*(1-math.exp(-18)))
    check('能量','RC供能',integral(lambda t:E*I(t),0,T),C/2*((E+(V-E)*math.exp(-9))**2-V**2)+integral(lambda t:R*I(t)**2,0,T))
for C1,C2,V1,V2 in [(1,2,3,-1),(2,2,2,0),(.1,3,-4,2)]:
    ce=C1*C2/(C1+C2);vf=(C1*V1+C2*V2)/(C1+C2)
    check('电荷与能量','交换差模热',ce*(V1-V2)**2/2,(C1*V1**2+C2*V2**2-(C1+C2)*vf**2)/2)
    for t in [.3,2]:
        d=(V1-V2)*math.exp(-.3*t)*(math.cos(math.sqrt(.91)*t)+.3/math.sqrt(.91)*math.sin(math.sqrt(.91)*t));u1=vf+C2*d/(C1+C2);u2=vf-C1*d/(C1+C2)
        check('电荷与能量','两电容共同电荷',C1*u1+C2*u2,C1*V1+C2*V2)
for z,T,h in [(.2,2,.6),(.6,4,1),(1,2,1),(2,4,.3)]:
    xe=np.array([1.,0]);d=phi(T-h,z)@(np.eye(2)-phi(h,z))@xe
    xs=np.linalg.solve(np.eye(2)-phi(T,z),d);xh=xe+phi(h,z)@(xs-xe)
    check('独立微分方程','周期不动点',rk(rk(xs,h,z,1),T-h,z,0),xs)
    heat1=integral(lambda t:np.array([2*z*(xe+phi(s,z)@(xs-xe))[1]**2 for s in t]),0,h)
    heat2=integral(lambda t:np.array([2*z*(phi(s,z)@xh)[1]**2 for s in t]),0,T-h)
    check('独立积分','周期供能',heat1+heat2,xh[0]-xs[0])
check('有效值积分','三角波',integral(lambda t:(4*t)**2,0,.25)*4,1/3)
check('有效值积分','半波',integral(lambda x:np.sin(x)**2,0,np.pi)/(2*np.pi),.25)
check('有效值积分','谐波',integral(lambda t:(2+3*np.cos(t)+4*np.sin(2*t))**2,0,2*np.pi)/(2*np.pi),16.5)
check('数字代入','W1过零',.1*math.log(4/3),.0287682072)
check('数字代入','W4峰值',math.exp(-.6*math.atan(.8/.6)/.8),.4988,tol=5e-5)
check('数字代入','W5剩余能量',5*math.exp(-2),.6767,tol=5e-5)
tp=math.atan(math.sqrt(96)/2)/math.sqrt(96)
check('数字代入','W8峰时',tp,.13977,tol=1e-5)
check('数字代入','W8校正',math.exp(2*tp),1.3225,tol=5e-5)
check('数字代入','W10相角',np.angle(-.5+1j*3*math.sqrt(3)/2)*180/np.pi,100.893,tol=1e-5)
check('数字代入','T7超调',math.exp(-math.pi*.3/math.sqrt(.91)),.3723,tol=5e-5)
check('数字代入','T7阻尼界',math.log(20)/math.sqrt(math.pi**2+math.log(20)**2),.6901,tol=5e-5)
check('相量','串联电压',3*(1.2+1.6j)-4j*(1.2+1.6j),10)
check('相量','并联电流',12/6+12/(4j)+12/(-3j),2+1j)
check('相量','混合输入',1/(1/(1+2j)+1j),.5-1.5j)
for a,V in [(1,0),(.2,3),(4,-2)]:
    up0=1/(1+a*a);d=V-up0;t=.7
    # Differential residual by fourth-order finite difference.
    f=lambda t:(math.cos(a*t)+a*math.sin(a*t))/(1+a*a)+d*math.exp(-t)
    h=1e-4;der=(f(t-2*h)-8*f(t-h)+8*f(t+h)-f(t+2*h))/(12*h)
    check('有限差分','正弦RC完整解',der+f(t),math.cos(a*t))
groups={}
for q in records:
    g=groups.setdefault(q['group'],dict(count=0,max_error=0));g['count']+=1;g['max_error']=max(g['max_error'],q['normalized_error'])
report=dict(count=len(records),count_description='独立比较的参数实例数；不是独立公式数量',all_passed=all(q['passed'] for q in records),groups=groups,records=records)
p=Path(__file__).resolve().parent/'检查/第二十一讲_公式核验.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False));assert report['all_passed'],[q for q in records if not q['passed']]
