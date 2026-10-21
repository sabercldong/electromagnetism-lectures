from pathlib import Path
from collections import Counter
import json,math
import numpy as np

ROOT=Path(__file__).resolve().parent
rows=[]
gx,gw=np.polynomial.legendre.leggauss(256)
def quad(fn,a,b):
    x=(b-a)*gx/2+(a+b)/2
    return float((b-a)/2*np.dot(gw,fn(x)))
def check(name,actual,expected,tol=2e-8,kind='积分或恒等式'):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    scale=float(np.max(np.abs(b)));error=float(np.max(np.abs(a-b)))/(scale if scale>1e-14 else 1.)
    rows.append(dict(name=name,normalized_error=error,tolerance=tol,kind=kind))
    assert np.isfinite(error) and error<=tol,(name,error,tol,a,b)
def rk(fn,y0,T,n=5000):
    y=np.asarray(y0,dtype=float);h=T/n
    for j in range(n):
        t=j*h;k1=fn(t,y);k2=fn(t+h/2,y+h*k1/2)
        k3=fn(t+h/2,y+h*k2/2);k4=fn(t+h,y+h*k3)
        y+=h*(k1+2*k2+2*k3+k4)/6
    return y
pi=math.pi

# 单位化eps0=1；零期望值的误差采用这一单位化条件的绝对误差。
for k in [1,2,4,8,20]:
    Q=.8;S=1.3;d=.7;U=1.1;C0=S/d;C=k*C0
    E=Q/(k*S);P=(k-1)*E
    check('固定电荷的界面总电荷',Q-P*S,Q/k)
    check('固定电荷的体积分场能',quad(lambda z:np.full_like(z,.5*k*E*E*S),0,d),Q*Q/(2*C))
    check('固定电压的总场',C*U/(k*S),U/d)
    check('恒压填充电源供能',U*(C-C0)*U,.5*(C-C0)*U*U+.5*(C-C0)*U*U)
for ks,ds,U in [([2,4,8],[.25,.5,.25],1),([3,6],[.4,.8],2),([1.5,7],[.9,.3],.7)]:
    ks=np.array(ks);ds=np.array(ds);D=U/sum(ds/ks);E=D/ks;P=D-E
    check('平板串联电压',ds@E,U)
    check('平板串联积分能量',.5*sum(ds*ks*E**2),.5*D*U)
    charges=np.r_[-P[0],P[:-1]-P[1:],P[-1]]
    check('多层平板总极化电荷',sum(charges),0)
    check('多层平板总电荷跳变',np.diff(E),P[:-1]-P[1:])
D=32/9
check('三层平板场与电压份额',np.r_[D/np.array([2,4,8]),D/np.array([2,4,8])*[.25,.5,.25]],[16/9,8/9,4/9,4/9,4/9,1/9])
for L,h,E0 in [(1,.7,2),(2,1.1,.8),(.6,2,1.5)]:
    line=quad(lambda x:np.full_like(x,2*E0),0,L)+quad(lambda x:np.full_like(x,3*E0),L,0)
    area=quad(lambda y:np.full_like(y,-E0*L/h),0,h)
    check('电位移回路直接积分',line,-E0*L)
    check('电位移斯托克斯面积分',area,line)
    delta=1e-5
    eps=lambda y:2+y/h
    check('电位移旋度有限差分',-E0*(eps(.4*h+delta)-eps(.4*h-delta))/(2*delta),-E0/h,kind='有限差分')

# 径向分层的电容、电势与场能分别直接积分。
for a,b,k in [(.8,2.5,2),(1,3,4),(1.3,2.7,8)]:
    q=.7;r=.6*a
    react=q/(4*pi)*(1/k-1)*(1/a-1/b)
    phi=quad(lambda x:q/(4*pi*x*x),r,a)+quad(lambda x:q/(4*pi*k*x*x),a,b)+q/(4*pi*b)
    check('点电荷球壳内反应电势',phi-q/(4*pi*r),react)
    check('球壳极化表面总电荷',-(1-1/k)*q+(1-1/k)*q,0)
    energy_diff=quad(lambda x:(1/k-1)*q*q/(8*pi*x*x),a,b)
    check('球壳反应能积分',energy_diff,.5*q*react)
for a,R,b,k1,k2 in [(.7,1.3,2.4,2,4),(1,1.5,3,6,2),(1.2,1.8,3.1,8,3)]:
    Q=.9;K=(1/a-1/R)/k1+(1/R-1/b)/k2
    voltage=quad(lambda r:Q/(4*pi*k1*r*r),a,R)+quad(lambda r:Q/(4*pi*k2*r*r),R,b)
    energy=quad(lambda r:Q*Q/(8*pi*k1*r*r),a,R)+quad(lambda r:Q*Q/(8*pi*k2*r*r),R,b)
    check('双层球形电势积分',voltage,Q*K/(4*pi))
    check('双层球形场能积分',energy,Q*voltage/2)
    check('球形界面极化跳变',Q/(4*pi*R*R)*(1/k2-1/k1),Q/(4*pi*k2*R*R)-Q/(4*pi*k1*R*R))
    Vc=quad(lambda r:Q/(2*pi*k1*r),a,R)+quad(lambda r:Q/(2*pi*k2*r),R,b)
    check('双层同轴电势积分',Vc,Q/(2*pi)*(math.log(R/a)/k1+math.log(b/R)/k2))
    energyc=quad(lambda r:Q*Q/(4*pi*k1*r),a,R)+quad(lambda r:Q*Q/(4*pi*k2*r),R,b)
    check('双层同轴场能积分',energyc,Q*Vc/2)
for k,f in [(2,.5),(5,.25),(8,.7),(1,.3)]:
    # 球帽cos(theta)=1-2f，独立角积分取得材料加权立体角。
    c=1-2*f
    eps_integral=2*pi*(quad(lambda u:np.full_like(u,k),c,1)+quad(lambda u:np.ones_like(u),-1,c))
    check('角向介质的加权立体角',eps_integral,4*pi*(1+f*(k-1)))
    check('角向介质自由电荷份额',2*pi*k*(1-c)/eps_integral,k*f/(1+f*(k-1)))
check('锥形分区题给电容和电荷份额',[4*pi*(1+.25*4),5*.25/(1+.25*4)],[8*pi,5/8])
for k,rho,a in [(2,.7,.8),(4,1,1.3),(8,.3,2)]:
    total=quad(lambda r:4*pi*r*r*rho,0,a)
    vol=quad(lambda r:-4*pi*r*r*(1-1/k)*rho,0,a)
    surf=4*pi*a*a*(1-1/k)*rho*a/3
    check('带体电荷球自由电荷积分',total,4*pi*rho*a**3/3)
    check('带体电荷球极化总量',vol+surf,0)
    check('带体电荷球心电势积分',quad(lambda r:rho*r/(3*k),0,a)+total/(4*pi*a),rho*a*a/3+rho*a*a/(6*k))
check('三层球形电容及阈值',[(1-2/3)/2+(2/3-1/2)/5+(1/2-1/3)/3,2*3,5*(1.5**2)*1.5,3*4*4],[23/90,6,135/8,48])
check('三层球形最大电压及其余局部场',[6*23/90,6/(5*1.5**2),6/(3*2**2)],[23/15,8/15,.5])
for a,b,ea,lam in [(1,3,5,.8),(.7,2,8,1.2),(1.2,3,7,.6)]:
    E=lam/(2*pi*ea*a);P=lambda r:lam/(2*pi*r)-E
    delta=1e-5;r=(a+b)/2
    rho=-((r+delta)*P(r+delta)-(r-delta)*P(r-delta))/(2*delta*r)
    check('渐变同轴极化体电荷散度',rho,lam/(2*pi*ea*a*r),kind='有限差分')
    vol=quad(lambda r:np.full_like(r,lam/(ea*a)),a,b)
    surf=-2*pi*a*P(a)+2*pi*b*P(b)
    check('渐变同轴极化体面总电荷',vol+surf,0)
    check('渐变同轴电压积分',quad(lambda r:np.full_like(r,E),a,b),lam*(b-a)/(2*pi*ea*a))
check('两材料几何相同最大值比',[8/2,1/3,(8*1)/(2*3),(8*1)/(2*9)],[4,1/3,4/3,4/9])

# 球腔的系数从电势及法向D矩阵解出，另遍历全外域最大场。
for k in [1,1.5,2,4,6,20,100]:
    A,B=np.linalg.solve([[1,1],[1,-2*k]],[1,k])
    f=3*k/(1+2*k);beta=(1-k)/(1+2*k)
    check('球腔边界矩阵系数',[A,B],[f,beta],kind='边界线性方程')
    angles=np.linspace(0,pi,401)[:,None];t=np.linspace(0,1,401)[None,:]
    mag=np.sqrt((1+2*B*t)**2*np.cos(angles)**2+(1-B*t)**2*np.sin(angles)**2)
    check('球腔全外域采样最大场',mag.max(),f,kind='极值采样')
    check('球腔介质侧极点法向D连续',k*(1+2*B),A)
    check('球腔介质侧赤道切向E连续',1-B,A)
check('四倍介电球腔远近位置场',[4/3,1/3,4/3,1-2/(3*8),1+1/(3*8)],[4/3,1/3,4/3,11/12,25/24])
for k in [2,4,6,20]:
    for N in [.05,1/3,.90,0,1]:
        # 以周围介质为背景，E=1-N*(epsc/epsh-1)E。
        E=np.linalg.solve([[1+N*(1/k-1)]],[1])[0]
        check('椭球腔自洽线性方程',E,k/(k+N*(1-k)),kind='边界线性方程')
check('六倍介电椭球腔数值',[6/(6-5*.05),6/(6-5/3),6/(6-5*.9)],[24/23,18/13,4])
Ld=1/20+19/80
check('平面气隙数值',[1/Ld,1/(4*Ld),(1/20)/Ld,(19/20)/(4*Ld),1/Ld/4],[80/23,20/23,4/23,19/23,20/23])
E1,E2=np.linalg.solve([[4,-2],[.5,.5]],[-5,1])
check('陷获面电荷边界方程',[E1,E2],[-1/6,13/6],kind='边界线性方程')
check('陷获面电荷的总面电荷',5+3*E1-E2,E2-E1)
check('陷获面电荷与电极电荷总量',4*E1-2*E2+5,0)

# 欧姆迁移ODE与热积分；孤立球的边界收到电荷而非流出空间。
for eps,gamma,U0,T in [(2,.3,1,4),(4,.8,.7,2),(8,.6,1.3,3)]:
    tau=eps/gamma
    def flow(t,y):
        U,H=y;return np.array([-gamma/eps*U,gamma*U*U])
    U,H=rk(flow,[U0,0],T)
    check('孤立漏电容电压ODE',U,U0*math.exp(-T/tau),kind='常微分方程积分')
    check('孤立漏电容热量ODE',H,.5*eps*U0*U0*(1-math.exp(-2*T/tau)),kind='常微分方程积分')
    a=.9;rho0=.8
    def sphereflow(t,y):
        rho,sigma,H=y
        power=quad(lambda r:4*pi*r*r*gamma*(rho*r/(3*eps))**2,0,a)
        return np.array([-rho/tau,gamma*rho*a/(3*eps),power])
    rho,sigma,H=rk(sphereflow,[rho0,0,0],T,n=1200)
    check('漏电球体面电荷ODE',[rho,sigma],[rho0*math.exp(-T/tau),rho0*a/3*(1-math.exp(-T/tau))],kind='常微分方程积分')
    check('漏电球总自由电荷',4*pi*a**3*rho/3+4*pi*a*a*sigma,4*pi*a**3*rho0/3)
    W0=2*pi*rho0*rho0*a**5/(45*eps)
    check('漏电球焦耳积分',H,W0*(1-math.exp(-2*T/tau)),kind='常微分方程积分')

# 空间电荷的高斯积分、势差、平方场能和机械力。
for n in [0,.5,1,1.5,2,5,12]:
    Sigma=.7;delta=1.2;eps=3
    rho=lambda x:-(n+1)*Sigma*x**n/delta**(n+1)
    for frac in [.2,.6,1]:
        x=frac*delta;Dn=Sigma+quad(rho,0,x)
        exact=Sigma*(1-frac**(n+1))
        check('幂律空间电荷的高斯积分',Dn,exact,tol=2e-8)
    V=quad(lambda x:Sigma/eps*(1-(x/delta)**(n+1)),0,delta)
    W=quad(lambda x:Sigma*Sigma/(2*eps)*(1-(x/delta)**(n+1))**2,0,delta)
    check('幂律空间电荷电势积分',V,Sigma*delta/eps*(n+1)/(n+2))
    check('幂律空间电荷场能积分',W,Sigma*Sigma*delta/(2*eps)*(1-2/(n+2)+1/(2*n+3)))
for eps1,eps2,d1,d2 in [(2,4,.6,.8),(6,2,.3,1.1),(3,3,.8,.4)]:
    Sigma=.9;L=d1/eps1+d2/eps2
    V=quad(lambda x:Sigma*(x+d1)/(d1*eps1),-d1,0)+quad(lambda x:Sigma*(1-x/d2)/eps2,0,d2)
    W=quad(lambda x:Sigma**2*((x+d1)/d1)**2/(2*eps1),-d1,0)+quad(lambda x:Sigma**2*(1-x/d2)**2/(2*eps2),0,d2)
    check('双侧接触层电势积分',V,Sigma*L/2)
    check('双侧接触层场能积分',W,Sigma*L/3*Sigma/2)
    rhob1=-(1-1/eps1)*Sigma/d1;rhob2=(1-1/eps2)*Sigma/d2
    bound_interface=Sigma*(1/eps2-1/eps1)
    check('双侧接触层极化总电荷',rhob1*d1+rhob2*d2+bound_interface,0)
    s=.7;h=1e-5;energy=lambda s:Sigma*Sigma*L/6+Sigma*Sigma*s/2
    pressure=(energy(s+h)-energy(s-h))/(2*h)
    check('冻结接触层机械压力',pressure,Sigma*Sigma/2,kind='有限差分')
    check('慢平衡分离的电势保持',(2*V/L)/(1+2*s/L)*(L/2+s),V)
for A,B,eta in [(2,3,.01),(1.5,.8,.1),(.7,2,.3)]:
    Leta=math.log(1+1/eta);xm=math.e*Leta/A;Vm=B*xm
    V=lambda x:B*x/math.log(A*x/Leta);h=xm*1e-5
    check('帕邢关系最小电压',V(xm),Vm)
    check('帕邢最小点导数',(V(xm+h)-V(xm-h))/(2*h),0,tol=2e-8,kind='有限差分')
    assert V(xm*.8)>Vm and V(xm*1.2)>Vm
    check('汤森电极反馈关系',eta*(math.exp(Leta)-1),1)

# 指定分离运动的RC模型同时积分电荷、热、机械功。
for l0,v,R,K,T in [(1,.7,8,1.2,3),(.8,.2,1,2,2),(1.3,1.2,4,.8,2)]:
    Q0=.9
    def separating(t,y):
        Q,H,A=y;l=l0+v*t;U=Q*l/K
        return np.array([-U/R,U*U/R,Q*Q*v/(2*K)])
    Q,H,A=rk(separating,[Q0,0,0],T)
    Qa=Q0*math.exp(-(l0*T+v*T*T/2)/(R*K))
    check('分离RC电荷ODE',Q,Qa,kind='常微分方程积分')
    W0=Q0*Q0*l0/(2*K);W=Q*Q*(l0+v*T)/(2*K)
    check('分离RC积分能量',W+H,W0+A,kind='常微分方程积分')
    if R*K*v>l0*l0:
        tp=(math.sqrt(R*K*v)-l0)/v
        U=lambda t:Q0*(l0+v*t)/K*math.exp(-(l0*t+v*t*t/2)/(R*K))
        h=1e-5
        check('分离RC峰值导数',(U(tp+h)-U(tp-h))/(2*h),0,kind='有限差分')
for e1,e2,h in [(2,5,.8),(5,2,1.3),(3,3,.9),(1,100,1.2)]:
    q=.7
    qp,qpp=np.linalg.solve([[1/e1,-1/e2],[-1,-1]],[-q/e1,-q])
    check('介质像法系数矩阵',[qp,qpp],[q*(e1-e2)/(e1+e2),2*e2*q/(e1+e2)],kind='边界线性方程')
    for rho in [0,.5,1.4]:
        dist=math.sqrt(rho*rho+h*h)
        check('介质像法平面电势连续',(q+qp)/(4*pi*e1*dist),qpp/(4*pi*e2*dist))
        check('介质像法平面法向D连续',h*(-q+qp)/(4*pi*dist**3),-h*qpp/(4*pi*dist**3))
    delta=1e-5;react=lambda z:q*qp/(16*pi*e1*z)
    F=-(react(h+delta)-react(h-delta))/(2*delta)
    check('介质像法反应能导数',F,q*qp/(16*pi*e1*h*h),kind='有限差分')

# 综合17：直接径向积分，再采样整个允许设计区间，而非只平衡阈值。
opt=math.sqrt(2)
for R in [1.05,1.3,opt,1.8,2.5,2.95]:
    K=quad(lambda r:1/(6*r*r),1,R)+quad(lambda r:1/(2*r*r),R,3)
    C=4*pi/K;Qm=min(48*pi,24*pi*R*R)
    check('耐压设计电容径向积分',C,12*pi*R)
    U=Qm/C;piece=2*R if R<=opt else 4/R
    check('耐压设计分段上限',U,piece)
Rs=np.r_[np.linspace(1,3,5001),opt]
Us=np.where(Rs<=opt,2*Rs,4/Rs)
check('耐压设计全区间最大值',Us.max(),2*opt,kind='极值采样')
check('耐压设计最优场和能量',[48*pi/(4*pi*6),48*pi/(4*pi*2*opt**2),.5*12*pi*opt*(2*opt)**2],[2,3,48*pi*opt])

# 综合18：从总电流方程独立积分，检查界面连续方程和供能。
eps1=4;eps2=2;g1=1;g2=9;d1=d2=.5;U=1;tau=.6
for T in [.1,.2,1.,2.]:
    # 只以数学线性系统核验超过实际击穿时刻的参数解；物理讲义明确止于阈值。
    def layered(t,y):
        E1,H,Ws=y;E2=(U-d1*E1)/d2
        dot=(g2*U-(g1*d2+g2*d1)*E1)/(eps1*d2+eps2*d1)
        J=g1*E1+eps1*dot
        return np.array([dot,d1*g1*E1**2+d2*g2*E2**2,U*J])
    E1,H,Ws=rk(layered,[2/3,0,0],T)
    E2=2-E1;decay=math.exp(-T/tau)
    check('双层漏电场ODE',[E1,E2],[9/5-17/15*decay,1/5+17/15*decay],kind='常微分方程积分')
    sf=eps2*E2-eps1*E1
    check('双层漏电界面自由电荷',sf,-34/5*(1-decay))
    dotsf=-34/5/tau*decay
    check('双层漏电界面连续方程',g1*E1-g2*E2,dotsf)
    dot=(18-10*E1)/6;J=g1*E1+eps1*dot
    check('双层漏电总电流系数',J,9/5+289/45*decay)
    w=.5*(d1*eps1*E1**2+d2*eps2*E2**2)
    w0=.5*(d1*eps1*(2/3)**2+d2*eps2*(4/3)**2)
    check('双层漏电能量ODE',w-w0+H,Ws,kind='常微分方程积分')
    check('双层漏电电源供能解析积分',Ws,U*(9/5*T+289/45*tau*(1-decay)),kind='常微分方程积分')
    ph=d1*g1*E1**2+d2*g2*E2**2
    wdot=d1*eps1*E1*dot-d2*eps2*E2*dot
    check('双层漏电瞬时功率',U*J,ph+wdot)
tc=tau*math.log(17/12)
check('迟发击穿临界时刻场',9/5-17/15*math.exp(-tc/tau),1)
check('迟发击穿全时段电压上限',min(1/(9/5),2/(4/3)),5/9)
groups={k:dict(count=sum(r['kind']==k for r in rows),maximum_normalized_error=max(r['normalized_error'] for r in rows if r['kind']==k)) for k in Counter(r['kind'] for r in rows)}
result=dict(count=len(rows),all_passed=True,maximum_normalized_error=max(r['normalized_error'] for r in rows),kinds=dict(Counter(r['kind'] for r in rows)),groups=groups,checks=rows,count_description='同一物理关系在不同参数下的计算实例，包含积分、边界方程、有限差分、ODE与极值采样；不代表独立公式数量。',zero_error_description='期望值为零时使用eps0=1等单位化条件下的绝对误差。')
(ROOT/'检查'/'第十五讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
