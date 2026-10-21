from pathlib import Path
import math,json
import numpy as np
ROOT=Path(__file__).resolve().parent
nodes,weights=np.polynomial.legendre.leggauss(128)
checks=[]
def quad(f,a,b):
    return (b-a)/2*sum(w*f((a+b)/2+(b-a)/2*x) for x,w in zip(nodes,weights))
def d(f,x,h=2e-5):return (f(x+h)-f(x-h))/(2*h)
def ck(name,got,want,tol=3e-8,kind='积分或恒等式'):
    a=np.asarray(got,float);b=np.asarray(want,float)
    err=float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(b)))))
    checks.append(dict(name=name,normalized_error=err,tolerance=tol,kind=kind))
    assert err<tol,(name,err,tol)
def rk(f,y,tmax,dt=.001):
    y=np.array(y,float);n=math.ceil(tmax/dt);h=tmax/n
    for j in range(n):
        t=j*h;k1=f(t,y);k2=f(t+h/2,y+h*k1/2);k3=f(t+h/2,y+h*k2/2);k4=f(t+h,y+h*k3)
        y+=h*(k1+2*k2+2*k3+k4)/6
    return y
# 用势的空间差分和直接毕奥--萨伐尔叠加比较。
for gap in [.5,1.,2.]:
 for x,y in [(0,.3),(.2,.8),(-.4,1.1)]:
    def A(x,y):return .5*math.log(((x-gap)**2+y*y)/((x+gap)**2+y*y))
    rp=(x+gap)**2+y*y;rm=(x-gap)**2+y*y
    ck('反向直电流势旋度', [d(lambda s:A(x,s),y),-d(lambda s:A(s,y),x)],
       [y*(1/rm-1/rp),(x+gap)/rp-(x-gap)/rm],kind='有限差分与源场叠加')
# 不同外场和运动参数，用磁通数值导数检验两个电动势之和。
for a in [.4,1.3]:
 for v in [.2,1.1]:
  for lam in [-.7,.3,1.2]:
   for t in [.1,.8,2.]:
    x=1+v*t;Is=math.exp(lam*t);f=math.log((x+a)/x)
    flux=lambda s:math.exp(lam*s)*math.log((1+v*s+a)/(1+v*s))
    em=Is*v*a/(x*(x+a));ei=-lam*Is*f
    ck('移动矩形两项与磁通导数',em+ei,-d(flux,t),kind='有限差分')
    R=.8;Ic=(em+ei)/R;Fx=-Ic*Is*a/(x*(x+a))
    ck('移动矩形机械与感生供能',-Fx*v+Ic*ei,Ic*Ic*R)
# 二棒方程直接积分，包含热量和位置，不用指数解推进。
for m1,m2 in [(.6,1.4),(1,1),(2,.7)]:
 for u1,u2 in [(0,1.2),(.7,-.2),(.4,.4)]:
    k=.9;mr=m1*m2/(m1+m2);tau=mr/k;T=2.3*tau
    def f(t,s):
        v1,v2,Q,D=s;w=v2-v1
        return np.array([k*w/m1,-k*w/m2,k*w*w,w])
    num=rk(f,[u1,u2,0,2],T);w=(u2-u1)*math.exp(-T/tau);V=(m1*u1+m2*u2)/(m1+m2)
    exact=[V-m2*w/(m1+m2),V+m1*w/(m1+m2),mr*((u2-u1)**2-w*w)/2,2+(u2-u1)*tau*(1-math.exp(-T/tau))]
    ck('二棒速度位置与热量',num,exact,5e-10,'耦合运动方程积分')
# 曲线旋转积分：路径可径向折返，终值仍由端点决定。
for B in [-.8,.4,1.7]:
 for om in [-.6,1.1]:
    r=lambda s:1+.8*s+.3*math.sin(2*math.pi*s)
    rp=lambda s:.8+.6*math.pi*math.cos(2*math.pi*s)
    ck('弯曲旋转导体开口积分',quad(lambda s:om*B*r(s)*rp(s),0,1),om*B*(r(1)**2-r(0)**2)/2)
# 圆盘与非均匀旋转棒，独立积分转矩与电阻。
for a,b in [(.3,1.1),(.6,2.)]:
    sigma=1.3;h=.2;B=.8;om=1.2;Re=.7
    Rd=quad(lambda r:1/(2*math.pi*sigma*h*r),a,b)
    ck('环形圆盘径向电阻',Rd,math.log(b/a)/(2*math.pi*sigma*h))
    c=B*(b*b-a*a)/2;Ic=c*om/(Re+Rd)
    torque=quad(lambda r:-Ic*B*r,a,b)
    ck('环形圆盘力矩与功率',[-torque, -torque*om],[Ic*c,Ic*Ic*(Re+Rd)])
for a in [.5,1.,2.]:
    coeff=quad(lambda r:r*(1-r*r/a**2),0,a)
    ck('径向非均匀场旋转积分',coeff,a*a/4)
# 非均匀轴对称场：从源场径向积分计算E，再检查旋度。
for scale in [.6,1.4]:
 for r in [.15,.7,1.3,3.]:
    E=lambda r:-quad(lambda s:s*math.exp(-s*s/scale**2),0,r)/r
    exact=-scale**2*(1-math.exp(-r*r/scale**2))/(2*r)
    ck('高斯场感应积分',E(r),exact)
    ck('高斯场感应旋度',d(lambda s:s*E(s),r)/r,-math.exp(-r*r/scale**2),kind='有限差分')
u=1.25643120862617
ck('高斯场极值正根',math.exp(u),1+2*u,1e-13)
# 薄盘及磁通孔外环的真实体积分。
for a in [.4,1.,1.8]:
 for rate in [-.7,.2,1.3]:
    sigma=1.6;h=.15
    diskP=quad(lambda r:sigma*(r*rate/2)**2*2*math.pi*r*h,0,a)
    ck('薄盘焦耳功率体积分',diskP,math.pi*sigma*h*a**4*rate**2/8)
    b=2*a
    annP=quad(lambda r:sigma*(rate/(2*math.pi*r))**2*2*math.pi*r*h,a,b)
    ck('磁通孔外环焦耳积分',annP,sigma*h*rate**2*math.log(b/a)/(2*math.pi))
# 电容导轨直接积分三个微分变量，热量单独推进。
for C in [.2,.8,2.]:
 for mass in [.7,1.5]:
    c=.9;R=.6;v0=1.3;eta=C*c*c/mass;tau=R*C/(1+eta);T=3.1*tau
    def f(t,s):
        Q,v,heat=s;Ic=(c*v-Q/C)/R
        return np.array([Ic,-c*Ic/mass,R*Ic*Ic])
    num=rk(f,[0,v0,0],T)
    Q=C*c*v0/(1+eta)*(1-math.exp(-T/tau));v=v0/(1+eta)*(1+eta*math.exp(-T/tau))
    heat=mass*v0*v0/2-mass*v*v/2-Q*Q/(2*C)
    ck('电容导轨充电速度热量',num,[Q,v,heat],2e-9,'耦合电路运动方程积分')
    ck('电容导轨动量型守恒',mass*num[1]+c*num[0],mass*v0,1e-11)
# 下落导轨与有限磁场带，用运动和热量方程积分检验。
for mass in [.6,1.4]:
 for k in [.4,1.2]:
    g=1.3;tau=mass/k;T=1.7*tau
    num=rk(lambda t,s:np.array([s[1],g-k*s[1]/mass,k*s[1]**2]),[0,0,0],T)
    v=g*tau*(1-math.exp(-T/tau));y=g*tau*(T-tau*(1-math.exp(-T/tau)))
    ck('重力导轨位移速度发热',num,[y,v,mass*g*y-mass*v*v/2],2e-9,'运动热量方程积分')
mass=.7;k=1.1;a=.8;D=2*a;tau=mass/k
for factor in [2.5,3.5,5.]:
    v0=factor*k*a/mass;v1=v0-k*a/mass;v2=v1-k*a/mass
    tin=tau*math.log(v0/v1);tout=tau*math.log(v1/v2)
    f=lambda t,s:np.array([s[1],-k*s[1]/mass,k*s[1]**2])
    entered=rk(f,[0,v0,0],tin)
    ck('磁场带进入边界状态',entered,[a,v1,mass*(v0*v0-v1*v1)/2],2e-9,'分段运动方程积分')
    exited=rk(f,[D,entered[1],entered[2]],tout)
    ck('磁场带离开边界状态',exited,[D+a,v2,mass*(v0*v0-v2*v2)/2],2e-9,'分段运动方程积分')
v0=.6*k*a/mass
num=rk(lambda t,s:np.array([s[1],-k*s[1]/mass]),[0,v0],20*tau)
ck('磁场带不足穿过的停止极限',num,[mass*v0/k,0],3e-9,'长时间运动方程积分')
# 弹簧磁阻尼的三种阻尼状态。
for k in [.6,2.,3.4]:
    mass=1.;kap=1.;gamma=k/(2*mass);T=1.3;x0=.8;v0=-.2
    num=rk(lambda t,s:np.array([s[1],-(k*s[1]+kap*s[0])/mass,k*s[1]**2]),[x0,v0,0],T)
    if gamma<1:
        wd=math.sqrt(1-gamma*gamma);D0=(v0+gamma*x0)/wd
        x=math.exp(-gamma*T)*(x0*math.cos(wd*T)+D0*math.sin(wd*T))
        v=-gamma*x+math.exp(-gamma*T)*wd*(-x0*math.sin(wd*T)+D0*math.cos(wd*T))
    elif gamma==1:
        D0=v0+gamma*x0;x=(x0+D0*T)*math.exp(-gamma*T)
        v=D0*math.exp(-gamma*T)-gamma*x
    else:
        delta=math.sqrt(gamma*gamma-1);l1=-gamma+delta;l2=-gamma-delta
        C0=(v0-l2*x0)/(l1-l2);D0=x0-C0
        x=C0*math.exp(l1*T)+D0*math.exp(l2*T);v=l1*C0*math.exp(l1*T)+l2*D0*math.exp(l2*T)
    heat=(mass*v0*v0+kap*x0*x0-mass*v*v-kap*x*x)/2
    ck('弹簧磁阻尼的三个状态',num,[x,v,heat],3e-10,'运动热量方程积分')
# 圆弧轨道：直接对磁通差分，分别数值积分两种供能。
for beta in [.15,.6]:
 for omega in [.3,1.]:
    B0=.8;theta0=.25;rad=1.2;c=rad*rad/2;R=.7;T=.8
    A=B0*omega+beta*theta0;D=2*beta*omega
    Ic=lambda t:c*(A+D*t)/R
    flux=lambda t:-c*(B0+beta*t)*(theta0+omega*t)
    ck('圆弧回路动生感生与磁通导数',c*(A+D*.4),-d(flux,.4),kind='有限差分')
    Wm=quad(lambda t:Ic(t)*c*(B0+beta*t)*omega,0,T)
    Wi=quad(lambda t:Ic(t)*c*beta*(theta0+omega*t),0,T)
    heat=c*c/R*(A*A*T+A*D*T*T+D*D*T**3/3)
    ck('圆弧回路两种供能积分',Wm+Wi,heat)
    ck('圆弧回路传输电荷积分',quad(Ic,0,T),c/R*(A*T+D*T*T/2))
# 引线走向、膨胀圆环、拍频及规范变换。
for R1,R2 in [(.4,1.3),(1.,1.),(2.,.3)]:
    eps=.7;I=eps/(R1+R2);phi=eps/2-I*R1
    ck('电压引线上下读数差',-I*R1-I*R2,-eps)
    ck('标势加感应路径项', [phi-eps/2,phi+eps/2],[-I*R1,I*R2])
for alpha,lam in [(.3,.6),(.2,.7),(.7,.4)]:
    flux=lambda t:math.pi*.8**2*1.1*math.exp((2*alpha-lam)*t)
    ck('扩张圆环的全磁通导数',(lam-2*alpha)*flux(.5),-d(flux,.5),kind='有限差分')
for om,Om in [(.8,.8),(.8,0),(.3,1.1),(1.1,.3)]:
    for t in [.2,.9,1.7]:
        Psi=lambda t:math.cos(Om*t)*math.cos(om*t)
        E=((Om+om)*math.sin((Om+om)*t)+(Om-om)*math.sin((Om-om)*t))/2
        ck('旋转与振荡差频电动势',E,-d(Psi,t),kind='有限差分')
for x,y in [(.3,.7),(-.4,.9),(1.2,-.6)]:
    dotB=.8;dotc=-.7
    a=np.array([(dotB/2-dotc)*y,(-dotB/2-dotc)*x]);p=np.array([dotc*y,dotc*x])
    ck('时间规范的电场抵消',a+p,[dotB*y/2,-dotB*x/2])
report={'count':len(checks),'all_passed':True,'maximum_normalized_error':max(x['normalized_error'] for x in checks),
        'description':'参数实例下的源场比较、有限差分、数值积分与耦合方程积分；不把实例数当作独立物理定律数。','checks':checks}
(ROOT/'检查'/'第十二讲_公式核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))
