from pathlib import Path
import math,json,numpy as np
records=[]
def check(group,name,a,b,tol=2e-7,scale=None):
    av=np.asarray(a);bv=np.asarray(b)
    den=max(float(np.max(np.abs(bv))),1e-25) if scale is None else scale
    err=float(np.max(np.abs(av-bv)))/den
    records.append(dict(group=group,name=name,computed=av.tolist(),reference=bv.tolist(),normalizing_scale=den,normalized_error=err,tolerance=tol,passed=err<=tol))
g,w=np.polynomial.legendre.leggauss(160)
def integ(f,a,b):
    if a==b:return 0.
    x=(b-a)*g/2+(a+b)/2
    return (b-a)/2*np.dot(w,np.asarray(f(x)))
def derivative(f,x,h=1e-5):
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
def rk4(rhs,y,T,steps=5000):
    y=np.array(y,float);h=T/steps
    for _ in range(steps):
        a=np.asarray(rhs(y));b=np.asarray(rhs(y+h*a/2));c=np.asarray(rhs(y+h*b/2));d=np.asarray(rhs(y+h*c))
        y+=h*(a+2*b+2*c+d)/6
    return y
def j0(x):
    term=1.;s=term
    for n in range(1,80):
        term*=-(x*x/4)/(n*n);s+=term
        if abs(term)<1e-18:break
    return s
def j1(x):
    term=x/2;s=term
    for n in range(1,80):
        term*=-(x*x/4)/(n*(n+1));s+=term
        if abs(term)<1e-18:break
    return s
def bisect(f,a,b):
    fa=f(a)
    for _ in range(70):
        m=(a+b)/2
        if fa*f(m)>0:a=m;fa=f(a)
        else:b=m
    return (a+b)/2
for r in [.05,.2,.5,.8,1,1.5,3]:
    enclosed=integ(lambda s:2*np.pi*s,0,min(r,1))
    B=enclosed/(2*np.pi*r)
    check('独立面积积分','均匀圆板磁场',B,r/2 if r<=1 else 1/(2*r))
for a in [.1,.5,.8]:
    for r in [a/2,a,(1+a)/2,1,2]:
        enc=integ(lambda s:2*np.pi*s,a,min(r,1)) if r>a else 0.
        target=0 if r<=a else (r-a*a/r)/2 if r<=1 else (1-a*a)/(2*r)
        check('独立面积积分','环形极板磁场',enc/(2*np.pi*r),target,scale=1.)
for lam in [-.5,0,.5,2,5]:
    C=integ(lambda s:2*np.pi*s*(1+lam*s*s),0,1)
    check('独立面积积分','径向介电电容',C,np.pi*(1+lam/2))
    for r in [.1,.5,1,2]:
        flux=integ(lambda s:2*np.pi*s*(1+lam*s*s),0,min(r,1))
        B=flux/(2*np.pi*r)
        target=(r+lam*r**3/2)/2 if r<=1 else (1+lam/2)/(2*r)
        check('独立面积积分','径向介电磁场',B,target)
        if r<1:
            f=lambda x:(x+lam*x**3/2)*x/2
            jd=derivative(f,r)/r
            check('有限差分反演','由场反求介电',jd,1+lam*r*r)
for A in [1,5,20]:
    for r in [.2,.8,1.2,2,3]:
        flux=integ(lambda s:2*np.pi*s*(1+A*np.exp(-s*s)),0,r)
        f=r+A*(1-math.exp(-r*r))/r
        check('独立面积积分','指数介电剖面',flux/(2*np.pi*r),f/2)
ff=lambda u:u+20*(1-np.exp(-u*u))/u
up=bisect(lambda u:derivative(ff,u),1,1.5)
check('独立数值极值','内部峰位置',up,1.1907674694345172,tol=1e-8)
grid=np.linspace(.0001,3,200001)
check('独立数值极值','内部峰大小',max(ff(grid)),13.91843176909858,tol=1e-8)
check('独立数值极值','边缘场较低',ff(3),9.665843934638, tol=1e-10)
# Direct line integral on the off-centre loop, split at disk intersections.
def tangential(theta):
    xx=1+np.cos(theta);yy=np.sin(theta);rr=np.hypot(xx,yy)
    rr=np.maximum(rr,1e-15)
    mag=np.where(rr<=1,rr/2,1/(2*rr))
    bx=-mag*yy/rr;by=mag*xx/rr
    return bx*(-np.sin(theta))+by*np.cos(theta)
line=sum(integ(tangential,a,b) for a,b in [(0,2*np.pi/3),(2*np.pi/3,np.pi),(np.pi,4*np.pi/3),(4*np.pi/3,2*np.pi)])
check('独立线积分','偏心环流',line,np.pi*(2/3-np.sqrt(3)/(2*np.pi)))
for qdot,ia,ib in [(-.3,1.1,-.2),(.8,-1,.5),(0,.2,.1)]:
    # Solve the two conservation equations as a linear system.
    unknown=np.linalg.solve(np.array([[-1,0],[0,1]]),np.array([-qdot-ia,qdot+ib]))
    if2,id1=unknown
    check('独立方程求解','两曲面全电流',ia+id1,if2+ib,scale=1.)
for chi in [1,3,5]:
    for x in [.2,1,3,10]:
        E=lambda t:np.sqrt(2)*np.cos(x*t)
        P=lambda t:np.sqrt(2)*chi/(1+x*x)*(np.cos(x*t)+x*np.sin(x*t))
        Pd=lambda t:np.sqrt(2)*chi*x/(1+x*x)*(-np.sin(x*t)+x*np.cos(x*t))
        T=2*np.pi/x
        avg=integ(lambda t:E(t)*Pd(t),0,T)/T
        heat=integ(lambda t:Pd(t)**2/chi,0,T)/T
        check('独立周期积分','极化平均损耗',avg,chi*x*x/(1+x*x))
        check('独立周期积分','极化耗散分解',heat,avg)
        initial=[P(0),0.,0.,0.]
        # Time is an extra state, avoiding reuse of the analytic derivative.
        def rhs(y):
            p,qh,ws,t=y;pd=chi*E(t)-p
            return [pd,pd*pd/chi,E(t)*pd,1.]
        end=rk4(rhs,initial,T)
        check('独立弛豫积分','稳态极化周期',end[0],P(0),tol=2e-6)
        check('独立弛豫积分','源功与热',end[2]/T,avg,tol=2e-6)
        check('独立弛豫积分','内部能收支',end[2]-end[1],(end[0]**2-initial[0]**2)/(2*chi),scale=1.,tol=2e-6)
for a,b,eps,sigma in [(1,3,2,.3),(1,2,4,1),(.5,4,1,.7)]:
    Q=1.3;C=4*np.pi*eps/(1/a-1/b);tau=eps/sigma
    for t in [.2*tau,tau,2*tau]:
        q=Q*np.exp(-t/tau)
        W=integ(lambda r:(q/(4*np.pi*eps*r*r))**2/2*eps*4*np.pi*r*r,a,b)
        heat=integ(lambda r:sigma*(q/(4*np.pi*eps*r*r))**2*4*np.pi*r*r,a,b)
        check('独立空间积分','球形场能',W,q*q/(2*C))
        check('独立空间积分','球形漏电热',heat,2*W/tau)
for t in [.2,1,2]:
    Q=lambda u:np.exp(-u-u*u/2);sigma=lambda u:1+u
    q=Q(t)
    check('有限差分','时变电导电荷',derivative(Q,t),-sigma(t)*q)
    heat=integ(lambda u:(1+u)*Q(u)**2,0,t)
    check('独立时间积分','时变电导发热',heat,.5*(1-q*q))
for t in [.2,.7,2]:
    C=lambda u:1+u;U=2.
    src=U*U*derivative(C,t);we=.5*U*U*derivative(C,t)
    check('变化参数','介质插入能量',src-we,we)
    h=lambda u:1+u
    we=lambda u:.5*U*U/h(u)
    q=lambda u:U/h(u)
    source=U*derivative(q,t);F=U*U/(2*h(t)**2)
    check('变化参数','极板分离能量',derivative(we,t),source+F)
    check('变化参数','电源收到两倍机械功',-source,2*F)

# Bessel fields: the series is checked against differential equations and
# independently integrated Ampere flux, rather than against the same series.
for eta in [.05,.1,.2,.4]:
    for r in [.2,.6,1.]:
        E=lambda s:j0(eta*s)
        ep=derivative(E,r)
        epp=derivative(lambda s:derivative(E,s,h=2e-4),r,h=2e-4)
        check('有限差分场方程','贝塞尔径向方程',epp+ep/r,-eta**2*E(r),tol=2e-6)
        check('有限差分场方程','贝塞尔磁场相位系数',ep,-eta*j1(eta*r))
        flux=integ(lambda s:np.array([j0(eta*v) for v in s])*s,0,r)
        check('独立积分场方程','贝塞尔安培积分',flux,r*j1(eta*r)/eta)
    check('误差界','四阶电场展开',1-eta**2/4+eta**4/64,j0(eta),tol=eta**6/2000)
check('数值边界','轴上参照误差',j0(.2),.9900249722395764,tol=1e-12)
check('数值边界','磁场主项参照误差',2*j1(.2)/.2,.9950083263923599,tol=1e-12)
limit=bisect(lambda eta:1/j0(eta)-1.01,.19,.21)
check('数值边界','给出的充分界确实小于1%',int(1/j0(.199)-1<.01),1,tol=0)
assert .199 < limit < .2

eps=2.;mu=3.;omega=1.7;k=omega*np.sqrt(eps*mu);E0=1.3
for x,t in [(.1,.2),(.3,.6),(.7,1.1),(.91,.45)]:
    for standing in [False,True]:
        if standing:
            E=lambda x,t:E0*np.cos(k*x)*np.cos(omega*t)
            B=lambda x,t:k/omega*E0*np.sin(k*x)*np.sin(omega*t)
        else:
            E=lambda x,t:E0*np.cos(k*x-omega*t)
            B=lambda x,t:k/omega*E(x,t)
        U=lambda x,t:(eps*E(x,t)**2+B(x,t)**2/mu)/2
        S=lambda x,t:E(x,t)*B(x,t)/mu
        group='有限差分驻波' if standing else '有限差分行波'
        check(group,'法拉第方程',derivative(lambda z:E(z,t),x),-derivative(lambda v:B(x,v),t))
        check(group,'安培方程',-derivative(lambda z:B(z,t),x)/mu,eps*derivative(lambda v:E(x,v),t))
        check(group,'局部能量收支',derivative(lambda v:U(x,v),t)+derivative(lambda z:S(z,t),x),0,scale=1.)
    Ex=lambda x,t:E0*np.cos(k*x)*np.cos(omega*t)
    rho=lambda x,t:-eps*k*E0*np.sin(k*x)*np.cos(omega*t)
    J=lambda x,t:eps*omega*E0*np.cos(k*x)*np.sin(omega*t)
    check('有限差分纵向场','高斯约束',eps*derivative(lambda z:Ex(z,t),x),rho(x,t))
    check('有限差分纵向场','电荷连续性',derivative(lambda v:rho(x,v),t)+derivative(lambda z:J(z,t),x),0,scale=1.)
    check('有限差分纵向场','电能与源功',derivative(lambda v:eps*Ex(x,v)**2/2,t),-Ex(x,t)*J(x,t))
    alpha=1.3
    Ey=lambda x,t:alpha*x*t
    Bz=lambda x,t:-alpha*t*t/2
    u=lambda x,t:(eps*Ey(x,t)**2+Bz(x,t)**2/mu)/2
    s=lambda x,t:Ey(x,t)*Bz(x,t)/mu
    jf=-eps*alpha*x
    check('有限差分给定源','源驱动场法拉第方程',derivative(lambda z:Ey(z,t),x),-derivative(lambda v:Bz(x,v),t))
    check('有限差分给定源','源驱动场能量',derivative(lambda v:u(x,v),t)+derivative(lambda z:s(z,t),x),-Ey(x,t)*jf)
check('独立周期积分','驻波平均能流',integ(lambda t:np.sin(2*omega*t),0,2*np.pi/omega),0,scale=1.)

# Solve the coupled Maxwell mode equations with heat as a third state.
for alpha in [0,.3,1.,2.]:
    def amplitude(t):
        if alpha<1:
            wd=np.sqrt(1-alpha*alpha)
            return np.exp(-alpha*t)*(np.cos(wd*t)-alpha/wd*np.sin(wd*t))
        if alpha==1:return (1-t)*np.exp(-t)
        beta=np.sqrt(alpha*alpha-1)
        return np.exp(-alpha*t)*(np.cosh(beta*t)-alpha/beta*np.sinh(beta*t))
    check('独立初值求导','导电模式初始斜率',derivative(amplitude,0),-2*alpha,scale=1.)
    for T in [.7,2.]:
        rhs=lambda y:np.array([y[1]-2*alpha*y[0],-y[0],alpha*y[0]**2])
        e,b,heat=rk4(rhs,[1.,0.,0.],T)
        check('独立耦合方程积分','导电模式电场',e,amplitude(T),scale=1.)
        check('独立耦合方程积分','导电模式磁场',b,derivative(amplitude,T)+2*alpha*amplitude(T),scale=1.)
        check('独立耦合方程积分','导电模式热与场能',heat+(e*e+b*b)/4,.25)
        power=integ(lambda x:(e*np.sin(x))**2*2*alpha,0,2*np.pi)/(2*np.pi)
        check('独立空间积分','导电模式平均热功率',power,alpha*e*e,scale=1.)

for z in [.1,1.,2.5]:
    a=1.;b=2.;ell=3.;eps=2.;U=1.3;du=.7
    cprime=2*np.pi*eps/np.log(b/a)
    iz=lambda q:cprime*du*(ell-q)
    flow=integ(lambda r:U/(r*np.log(b/a))*eps*du*(ell-z)/(r*np.log(b/a))*2*np.pi*r,a,b)
    we=integ(lambda r:eps/2*(U/(r*np.log(b/a)))**2*2*np.pi*r,a,b)
    check('独立同轴积分','横截面能流',flow,U*iz(z))
    check('独立同轴积分','每长度电场能',we,cprime*U*U/2)
    check('有限差分同轴守恒','轴向电流梯度',derivative(iz,z),-cprime*du)

for hall in [.5,2.,5.]:
    sigma=.8;eps=1.3
    for T in [.3,1.1,2.]:
        def rhs(y):
            ex,ey,heat=y
            return np.array([(-sigma*ex+hall*ey)/eps,(-sigma*ey-hall*ex)/eps,sigma*(ex*ex+ey*ey)])
        ex,ey,heat=rk4(rhs,[1.,0.,0.],T)
        amp=np.exp(-sigma*T/eps);angle=hall*T/eps
        check('独立横向耦合积分','横向电场旋转',np.array([ex,ey]),amp*np.array([np.cos(angle),-np.sin(angle)]))
        check('独立横向耦合积分','横向电流不做功',heat+eps*(ex*ex+ey*ey)/2,eps/2)

# RC finite-time energy integrals and the neglected magnetic energy bound.
for R,C in [(1,1),(2,.3),(.4,3)]:
    voltage=1.3;tau=R*C
    current=lambda t:voltage/R*np.exp(-t/tau)
    U=lambda t:voltage*(1-np.exp(-t/tau))
    for s in [.2,np.log(2),1.,3.]:
        T=s*tau
        src=integ(lambda t:voltage*current(t),0,T)
        heat=integ(lambda t:R*current(t)**2,0,T)
        energy=C*U(T)**2/2
        check('独立RC能量积分','电源有限时功',src,C*voltage**2*(1-np.exp(-s)))
        check('独立RC能量积分','电阻有限时热',heat,C*voltage**2/2*(1-np.exp(-2*s)))
        check('独立RC能量积分','有限时能量收支',src-heat,energy)
    pin=lambda t:U(t)*current(t)
    check('有限差分RC极值','侧面功率峰条件',derivative(pin,tau*np.log(2)),0,scale=1.)
    check('RC极值','侧面功率峰值',pin(tau*np.log(2)),voltage**2/(4*R))
for radius,h,mu,current in [(1,.2,3,.7),(.3,.1,2,1.3),(2,.05,1,.3)]:
    mag=integ(lambda r:mu*(current*r/(2*np.pi*radius**2))**2/2*2*np.pi*r*h,0,radius)
    check('独立磁能空间积分','充电磁能',mag,mu*h*current**2/(16*np.pi))

# The two leaky layers are integrated from equal D, using an interface
# voltage state and accumulated extra source work / heat above DC baseline.
for eps0,sigma,S,d,U in [(1,1,1,1,1),(2,.4,.7,.3,1.3),(.7,2,3,1.5,.8)]:
    tau=6*eps0/(5*sigma);sumE=U/d;F0=eps0*S*U*U/d
    e1inf=.8*sumE;e2inf=.2*sumE
    steadyJ=.8*sigma*sumE
    steadyHeat=S*d*sigma*(e1inf**2+4*e2inf**2)
    def rhs(y):
        e1,work,heat=y;e2=sumE-e1
        edot=(4*sigma*e2-sigma*e1)/(6*eps0)
        j=sigma*e1+2*eps0*edot
        return np.array([edot,S*U*(j-steadyJ),S*d*sigma*(e1*e1+4*e2*e2)-steadyHeat])
    initial=2*sumE/3
    energy=lambda e1:S*d*eps0*(e1*e1+2*(sumE-e1)**2)
    for s in [.3,1.,3.,12.]:
        e1,work,heat=rk4(rhs,[initial,0.,0.],s*tau)
        expected=sumE*(.8-2*np.exp(-s)/15)
        check('独立两层弛豫积分','两层场强响应',e1,expected)
        check('独立两层弛豫积分','额外源功',work,8/75*F0*(1-np.exp(-s)))
        check('独立两层弛豫积分','额外焦耳热',heat,4/75*F0*(1-np.exp(-2*s)))
        check('独立两层弛豫积分','有限时能量守恒',work-heat,energy(e1)-energy(initial))
        e2=sumE-e1;edot=rhs([e1,0,0])[0]
        charge=eps0*(4*e2-2*e1)
        check('两层界面守恒','界面自由电荷',charge,-.8*eps0*sumE*(1-np.exp(-s)))
        check('两层界面守恒','界面自由电流差',-6*eps0*edot,sigma*e1-4*sigma*e2)
        check('两层界面守恒','全电流连续',sigma*e1+2*eps0*edot,4*sigma*e2-4*eps0*edot)
    check('两层最终能量','稳态场能增量',energy(e1inf)-energy(initial),4/75*F0)

# Compare printed rounded values to fresh calculations in SI units.
eps0=8.8541878128e-12;mu0=4*np.pi*1e-7
R=.05;h=.001;C=eps0*np.pi*R*R/h
Z=np.sqrt(mu0/(4*eps0));omega=2*np.pi*1e6
rounded=[
 ('圆板总电流',eps0*np.pi*R*R*2e13,1.39081),
 ('圆板边缘磁场微特斯拉',mu0*eps0*R*2e13/2*1e6,5.56325),
 ('电容皮法',C*1e12,69.5406),
 ('恒流电压',.001*1e-5/C,143.801),
 ('恒流场能微焦',.001**2*1e-5**2/(2*C)*1e6,.719004),
 ('行波阻抗',Z,188.365),
 ('行波磁场幅值',60/Z,.318530),
 ('行波能流',60**2/(2*Z),9.55591),
 ('有损圆柱平均功率毫瓦',.02*100**2/2*np.pi*.01**2*.1*1e3,3.14159),
 ('RC峰时微秒',1e6*C*np.log(2)*1e6,48.2019),
 ('RC最终场能微焦',C*100**2/2*1e6,.347703),
 ('两层时间毫秒',6*eps0/(5*(eps0/.001))*1e3,1.2),
 ('两层界面电荷微库每平方米',-.8*eps0*100/.001*1e6,-.708335),
 ('两层初始电流微安',.01*(eps0/.001)*100/.001*(.8+4/45)*1e6,7.87039),
 ('两层最终电流微安',.01*(eps0/.001)*100/.001*.8*1e6,7.08335),
 ('两层场能增量纳焦',4/75*eps0*.01*100**2/.001*1e9,47.2223),
 ('两层额外源功纳焦',8/75*eps0*.01*100**2/.001*1e9,94.4447)]
for label,actual,printed in rounded:check('SI数值代入',label,actual,printed,tol=5e-6)
body=(Path(__file__).resolve().parents[1]/'src/lectures/第二十四讲_位移电流.tex').read_text()
for token in ['9.55591','0.990024972','0.995008326','1.01007550','47.2223','94.4447','69.5406']:
    assert token in body,token
groups={}
for r in records:groups[r['group']]=groups.get(r['group'],0)+1
report={'count':len(records),'count_description':'每次独立计算与一个解析结论的比较计一个参数实例；不是独立公式数量。','all_passed':all(r['passed'] for r in records),'groups':groups,'records':records}
out=Path(__file__).parent/'检查'/'第二十四讲_公式核验.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False))
for r in records:
    if not r['passed']:print(r)
assert report['all_passed']
