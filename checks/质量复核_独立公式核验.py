"""独立数值检查：直接库仑叠加、数值积分和有限差分对照讲义闭式结果。

电学检查采用一致的归一化单位 k=1、epsilon=1/(4*pi)，不把数值误差
当成物理测量不确定度。第四讲另运行原有的独立公式核验脚本。
"""
from pathlib import Path
import math
import json
import random
import statistics

P=math.pi
EPS=1/(4*P)
ROOT=Path(__file__).resolve().parent
records=[]
def check(label,actual,expected,tol=2e-7):
    err=abs(actual-expected)/max(1,abs(expected))
    records.append(dict(label=label,actual=actual,expected=expected,
                        scaled_error=err,tolerance=tol,passed=err<tol))
def integ(f,a,b,n=2400):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+sum((4 if j%2 else 2)*f(a+j*h) for j in range(1,n)))
def diff(f,x,h=1e-5):
    return (f(x+h)-f(x-h))/(2*h)
def field(sources,r):
    out=[0.,0.,0.]
    for q,s in sources:
        d=[r[i]-s[i] for i in range(3)]
        den=sum(x*x for x in d)**1.5
        for i in range(3):out[i]+=q*d[i]/den
    return out

# 第一讲：由点电荷原始叠加核对平衡与一阶微扰，而非重复抄公式。
for Q1,Q2,d in [(1,9,2.3),(2.4,.8,1.7)]:
    x0=d*math.sqrt(Q1)/(math.sqrt(Q1)+math.sqrt(Q2))
    src=[(Q1,(0,0,0)),(Q2,(d,0,0))]
    A=Q1/x0**3+Q2/(d-x0)**3
    check('第一讲 W1 平衡',field(src,(x0,0,0))[0],0)
    check('第一讲 W1 轴向线性系数',diff(lambda x:field(src,(x,0,0))[0],x0),-2*A)
    check('第一讲 W1 横向线性系数',diff(lambda y:field(src,(x0,y,0))[1],0),A)
for alpha in [.25,2.25,4]:
    x=2/(1-math.sqrt(alpha))
    check('第一讲 W2 异号平衡',field([(1,(0,0,0)),(-alpha,(2,0,0))],(x,0,0))[0],0)
check('第一讲 W4 中心负电荷失稳系数',
      -.25*diff(lambda x:field([(1,(-1,0,0)),(1,(1,0,0))],(x,0,0))[0],0),1)
for n in [3,4,5,9]:
    points=[(math.cos(2*P*j/n),math.sin(2*P*j/n),0) for j in range(n)]
    e=field([(1,p) for p in points[1:]],points[0])
    radial=sum(1/math.sin(P*j/n) for j in range(1,n))/4
    check(f'第一讲 W6 N={n} 径向叠加',e[0],radial)
    check(f'第一讲 W6 N={n} 切向相消',e[1],0)
L=1.8;kap=2.4;B=4*kap*L**3/27;xc=2*L/3
f=lambda x:-B/x**2-kap*(x-L)
check('第一讲 T11 临界平衡',f(xc),0)
check('第一讲 T11 临界二次系数',
      (f(xc+1e-4)+f(xc-1e-4)-2*f(xc))/(2e-8),-9*kap/(4*L),tol=2e-6)
for x in [.8,1.4]:
    q=-1.7*.6/(math.sqrt(1.7)+math.sqrt(.6))**2
    d=2.2;pos=d*math.sqrt(1.7)/(math.sqrt(1.7)+math.sqrt(.6))
    # 逐一求三个自由电荷的受力。
    src=[(1.7,(0,0,0)),(.6,(d,0,0)),(q,(pos,0,0))]
    j=0 if x<1 else 1
    check('第一讲 T10 两端平衡',src[j][0]*field([v for i,v in enumerate(src) if i!=j],src[j][1])[0],0)
A=4.04
check('第一讲 W8 指数估计',math.log(A)/math.log(2)-2,math.log(1.01)/math.log(2))
rng=random.Random(20261004)
samples=[math.log(rng.gauss(A,.01*A))/math.log(2)-2 for _ in range(100000)]
check('第一讲 W8 标准不确定度线性化与正态输入模拟',
      statistics.stdev(samples),.01/math.log(2),tol=3e-4)

# 第二讲：对线、环、面电荷的原始积分进行独立数值积分。
for r,z in [(1.1,.4),(2.3,-.7),(.8,2.1)]:
    z1,z2=-1.2,1.7
    er=integ(lambda u:r/(r*r+(z-u)**2)**1.5,z1,z2)
    ez=integ(lambda u:(z-u)/(r*r+(z-u)**2)**1.5,z1,z2)
    u1,u2=z1-z,z2-z
    check('第二讲 W1 有限棒径向',er,(u2/math.hypot(r,u2)-u1/math.hypot(r,u1))/r)
    check('第二讲 W1 有限棒轴向',ez,1/math.hypot(r,u2)-1/math.hypot(r,u1))
r=1.3;l=1.8;beta=.7
check('第二讲 W2 非均匀棒轴向',
      integ(lambda z:-(1+beta*z/l)*z/(r*r+z*z)**1.5,-l,l),
      -2*beta/l*(math.asinh(l/r)-l/math.hypot(r,l)))
check('第二讲 W3 非均匀完整圆环',
      integ(lambda t:-(1+.6*math.cos(t))*math.cos(t),0,2*P),-.6*P)
check('第二讲 W4 圆环平面内微扰',
      diff(lambda s:integ(lambda t:(s-math.cos(t))/(1+s*s-2*s*math.cos(t))**1.5,0,2*P),0),-P)
for a,z in [(1.4,.7),(.8,2.1),(2.3,-1.3)]:
    numerical=integ(lambda s:2*P*z*s/(z*z+s*s)**1.5,0,a)
    closed=2*P*(math.copysign(1,z)-z/math.hypot(z,a))
    check('第二讲 W5 圆盘轴线积分',numerical,closed)
for z in [-.7,0,.9]:
    check('第二讲 W6 带孔盘直接环积分',
          integ(lambda s:2*P*z*s/(z*z+s*s)**1.5,1,2),
          2*P*z*(1/math.hypot(z,1)-1/math.hypot(z,2)))
xp=2**(2/3)/math.sqrt(2**(2/3)+1)
check('第二讲 T9 环面最大值方程',
      diff(lambda x:x*(1/math.hypot(x,1)-1/math.hypot(x,2)),xp),0)
for a,b,z in [(1.2,.8,.7),(2,.6,1.3),(.7,1.8,2.1)]:
    num=4*integ(lambda x:integ(lambda y:z/(z*z+x*x+y*y)**1.5,0,b,n=120),0,a,n=120)
    check('第二讲 补充 矩形板双重积分',num,4*math.atan(a*b/(z*math.sqrt(z*z+a*a+b*b))),tol=2e-6)
for a,z in [(1,.6),(1.4,2.1),(.8,-1.2)]:
    num=4*integ(lambda x:z/(x*x+a*a+z*z)**1.5,-a,a)
    check('第二讲 补充 方形线框逐边积分',num,8*a*z/((a*a+z*z)*math.sqrt(2*a*a+z*z)))
x=2.1;sep=1e-3
exact=(1/(x+sep/2)**2-1/(x-sep/2)**2)/sep
check('第二讲 补充 非匀强场偶极合力',exact,-2/x**3,tol=2e-7)
check('第二讲 T16 三次恢复场',
      (2*P*.002*((1+.002**2)**-1.5-8*(4+.002**2)**-1.5))/(2*P*.002**3),
      -9/8,tol=1e-5)

# 第三讲：积分密度、从势求导、界面匹配及真实分段时间。
for n in [-1.5,0,1,2]:
    R=1.8;rho=1.3;r=.7
    # 以 u=t^(1/(n+3)) 消除原点的可积奇点。
    Qnum=4*P*rho*r**(n+3)/R**n*integ(lambda t:1/(n+3),0,1)
    E=rho*r**(n+1)/((n+3)*EPS*R**n)
    check(f'第三讲 W1 n={n} 包围电量',Qnum,4*P*EPS*r*r*E)
    phi=lambda s:rho*R*R/((n+3)*EPS)*(1+(1-(s/R)**(n+2))/(n+2))
    check(f'第三讲 W1 n={n} 势梯度',-diff(phi,r),E)
R=2.1;rho=1.3;s=.8
lam=2*P*integ(lambda u:rho*(1-u*u/(R*R))*u,0,s)
check('第三讲 W3 非均匀柱场',lam/(2*P*EPS*s),rho/(2*EPS)*(s-s**3/(2*R*R)))
smax=R*math.sqrt(2/3)
check('第三讲 W3 全局最大场',rho/(2*EPS)*(smax-smax**3/(2*R*R)),rho*R/(3*EPS)*math.sqrt(2/3))
for sig1,sig2 in [(1.1,-.7),(-.3,.8)]:
    a=1.4
    mid=lambda z:-(sig1-sig2)*z/(2*EPS)
    low=lambda z:((sig1+sig2)*z+2*a*sig1)/(2*EPS)
    high=lambda z:(-(sig1+sig2)*z+2*a*sig2)/(2*EPS)
    check('第三讲 W7 下边界势连续',low(-a),mid(-a))
    check('第三讲 W7 上边界势连续',high(a),mid(a))
for p in [1,2,2.7]:
    C=1.2;r=1.4;r0=.8
    phi=lambda s:C*math.log(r0/s) if p==1 else C/(p-1)*(s**(1-p)-r0**(1-p))
    check('第三讲 W9 径向势导数',-diff(phi,r),C/r**p)
rho=.9;a=1.2
phi=lambda z:rho/(2*EPS*a)*(a*a*z-z**3/3)
check('第三讲 T17 奇密度厚层势差',phi(a)-phi(-a),2*rho*a*a/(3*EPS))
check('第三讲 T17 从场反演密度',
      diff(lambda z:rho/(2*EPS*a)*(z*z-a*a),.4)*EPS,rho*.4/a)
# 时间积分作 r=2R*cos²theta 替换，消除静止释放点端点奇异性。
touter=integ(lambda theta:4*math.cos(theta)**2,0,P/4)
check('第三讲 T18 外部时间积分（omega=1）',touter,1+P/2)
check('第三讲 T18 全周期计段',4*touter+P,4+3*P)
result=dict(method='direct Coulomb sums, Simpson quadrature, finite differences, seeded normal-input simulation',
            normalization='k=1; epsilon=1/(4*pi); all lengths and charges use consistent normalized units',
            checks=records,count=len(records),passed=sum(r['passed'] for r in records))
(ROOT/'检查/质量复核_第一至三讲公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'count':len(records),'passed':result['passed'],
                  'failed':[r for r in records if not r['passed']]},ensure_ascii=False,indent=2))
assert result['passed']==len(records)
