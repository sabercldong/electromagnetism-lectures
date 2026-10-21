from pathlib import Path
import math,json
import numpy as np
ROOT=Path(__file__).resolve().parent
rows=[]
def check(name,actual,expected,tol=2e-7,kind='积分或恒等式'):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    error=float(np.max(np.abs(a-b))/max(float(np.max(np.abs(b))),1e-12))
    rows.append(dict(name=name,normalized_error=error,tolerance=tol,kind=kind))
    assert np.isfinite(error) and error<=tol,(name,error,tol,actual,expected)
gx,gw=np.polynomial.legendre.leggauss(256)
def quad(fn,a,b):
    x=(b-a)*gx/2+(a+b)/2
    return float((b-a)/2*np.dot(gw,fn(x)))
def rk(fn,y0,T,n=8000):
    y=np.array(y0,dtype=float);h=T/n
    for j in range(n):
        t=j*h;k1=fn(t,y);k2=fn(t+h/2,y+h*k1/2)
        k3=fn(t+h/2,y+h*k2/2);k4=fn(t+h,y+h*k3)
        y+=h*(k1+2*k2+2*k3+k4)/6
    return y

# 单位化e=m=c=1，从感应电场积分动量、功和圈数。
for R in [.6,1.4,2.2]:
    for eta in [.16,.36,.64]:
        Bi=.3;Bci=.9;beta=.7;kappa=(1+eta)*beta/eta;T=1.8
        a=R*math.sqrt(eta)
        avg0=(quad(lambda r:np.full_like(r,Bci)*2*r,0,a)+quad(lambda r:np.full_like(r,Bi)*2*r,a,R))/R**2
        avg1=(quad(lambda r:np.full_like(r,Bci+kappa*T)*2*r,0,a)+quad(lambda r:np.full_like(r,Bi+beta*T)*2*r,a,R))/R**2
        p0=R*Bi;pf=R*(Bi+beta*T)
        check('分区场磁通与动量增量',R*(avg1-avg0)/2,pf-p0)
        check('注入偏置与正则角动量',R*pf-R**2*avg1/2,R*p0-R**2*avg0/2)
        v=lambda t:R*(Bi+beta*t)/np.sqrt(1+R**2*(Bi+beta*t)**2)
        gain=quad(lambda t:R*beta*v(t),0,T)
        check('加速器电力功与相对论能量',gain,math.sqrt(1+pf**2)-math.sqrt(1+p0**2))
        turns=quad(v,0,T)/(2*math.pi*R)
        check('实际飞行圈数与每圈增量',2*math.pi*R**2*beta*turns,gain)

# 均匀缓变场的完整非相对论洛伦兹方程；不预设圆轨道收缩。
for rate in [.006,.01,.015]:
    T=1/rate
    def fn(t,y):
        x,z,vx,vz,work=y;B=1+rate*t
        Ex=rate*z/2;Ez=-rate*x/2
        ax=-Ex-vz*B;az=-Ez+vx*B
        return np.array([vx,vz,ax,az,-Ex*vx-Ez*vz])
    x,z,vx,vz,work=rk(fn,[1,0,0,1,0],T,n=24000)
    P=x*vz-z*vx-(1+rate*T)*(x*x+z*z)/2
    check('均匀时变场正则角动量',P,.5,tol=5e-8,kind='完整运动方程积分')
    check('均匀时变场电场功',work,(vx*vx+vz*vz-1)/2,tol=5e-8,kind='完整运动方程积分')
    check('均匀场缓变半径近似',math.hypot(x,z),1/math.sqrt(2),tol=2*rate,kind='缓变近似')

for a,b,h,N,I in [(1,2,.7,12,.8),(1.3,1.9,.4,20,1.2),(.8,3,1,7,2)]:
    mu=1.4;B=lambda r:mu*N*I/(2*math.pi*r)
    Psi=N*quad(lambda r:B(r)*h,a,b)
    energy=quad(lambda r:B(r)**2/(2*mu)*2*math.pi*r*h,a,b)
    L=mu*N*N*h/(2*math.pi)*math.log(b/a)
    check('环形线圈磁通链',Psi,L*I)
    check('环形线圈场能',energy,L*I*I/2)
for s in [0,.5,1,2,4,8,16,32]:
    a=1.3;I=.9;mu=1.7;C=(s+2)*I/(2*math.pi*a**(s+2))
    check('幂分布电流归一化',quad(lambda r:C*r**s*2*math.pi*r,0,a),I)
    B=lambda r:mu*I*(r/a)**(s+2)/(2*math.pi*r)
    energy=quad(lambda r:B(r)**2/(2*mu)*2*math.pi*r,0,a)
    check('幂分布内部电感',2*energy/(I*I),mu/(4*math.pi*(s+2)))
for b in [.8,1.3,2]:
    for ratio in [1.04,1.2,1.7,2.5]:
        c=b*ratio;mu=1.5
        numeric=mu/(2*math.pi)*quad(lambda r:((c*c-r*r)/(c*c-b*b))**2/r,b,c)
        exact=mu/(2*math.pi)*(c**4*math.log(c/b)-c*c*(c*c-b*b)+(c**4-b**4)/4)/(c*c-b*b)**2
        check('有限厚度同轴外导体电感',numeric,exact)
for eps in [.001,.003,.01]:
    b=1.1;c=b*(1+eps);mu=1
    exact=mu/(2*math.pi)*quad(lambda r:((c*c-r*r)/(c*c-b*b))**2/r,b,c)
    check('同轴薄返回壳首项',exact,mu*(c-b)/(6*math.pi*b),tol=2*eps,kind='渐近比较')

for L,R,U in [(1,.8,.7),(.6,1.5,1.3),(2.2,.4,1.7),(.9,2,.8)]:
    tau=L/R;T=2.3*tau
    def fn(t,y):
        i,heat,work=y
        return np.array([(U-R*i)/L,R*i*i,U*i])
    got=rk(fn,[0,0,0],T)
    If=U/R*(1-math.exp(-T/tau))
    Q=U*U/R*(T-2*tau*(1-math.exp(-T/tau))+tau/2*(1-math.exp(-2*T/tau)))
    Ws=U*U/R*(T-tau*(1-math.exp(-T/tau)))
    for k,value in enumerate([If,Q,Ws]):check('RL电流热量电源功',got[k],value,kind='回路方程积分')
    check('RL总能量',got[2]-got[1],L*got[0]**2/2,kind='回路方程积分')
for L,R,V,I0 in [(1,.8,.5,1.2),(2,1.5,.3,.8),(.5,.2,2,1.3),(.9,2,1,.7)]:
    T=L/R*math.log1p(R*I0/V)
    def fn(t,y):
        i,Q,W=y;return np.array([-(R*i+V)/L,R*i*i,V*i])
    i,Q,W=rk(fn,[I0,0,0],T)
    check('钳位停止电流',i+I0,I0,kind='回路方程积分')
    Wc=V/R*(L*I0-V*T)
    check('钳位器件能量',W,Wc,kind='回路方程积分')
    check('钳位放电总能量',Q+W,L*I0**2/2,kind='回路方程积分')

for L,M,R,I0 in [(1,.2,.8,1.1),(2,.7,1.2,.9),(.7,.5,1.5,1.2),(1.5,1,.5,.6)]:
    mat=np.array([[L,M],[M,L]]);inv=np.linalg.inv(mat)
    T=25*(L+M)/R
    def fn(t,y):
        curr=y[:2];dc=inv@(-R*curr)
        return np.array([dc[0],dc[1],R*curr[0]**2,R*curr[1]**2])
    got=rk(fn,[I0,0,0,0],T,n=16000)
    Q2=M*M*I0*I0/(4*L);Q1=L*I0*I0/2-Q2
    check('等阻耦合回路原热量',got[2],Q1,kind='耦合回路积分')
    check('等阻耦合回路次级热量',got[3],Q2,kind='耦合回路积分')
    t=.8*(L+M)/R
    yi=rk(fn,[I0,0,0,0],t)
    exact=I0/2*np.array([math.exp(-R*t/(L+M))+math.exp(-R*t/(L-M)),math.exp(-R*t/(L+M))-math.exp(-R*t/(L-M))])
    check('等阻耦合回路双指数',yi[:2],exact,kind='耦合回路积分')

for L,M,R,I0 in [(1,.2,.8,1.1),(2,.7,1.2,.9),(.7,.5,1.5,1.2),(1.5,1,.5,.6)]:
    mat=np.array([[L,M],[M,L]]);inv=np.linalg.inv(mat);D=L*L-M*M
    ls=R*(3*L-math.sqrt(L*L+8*M*M))/(2*D)
    lf=R*(3*L+math.sqrt(L*L+8*M*M))/(2*D)
    eig=np.sort(np.linalg.eigvals(inv@np.diag([R,2*R])))
    check('异阻耦合回路衰减率',[ls,lf],eig,kind='矩阵本征值')
    def fn(t,y):
        i=y[:2];d=inv@(-np.array([R,2*R])*i)
        return np.array([d[0],d[1],i[0],i[1],R*i[0]**2+2*R*i[1]**2])
    T=25/ls;got=rk(fn,[I0,0,0,0,0],T,n=16000)
    check('异阻耦合回路传输电荷',got[2:4],[L*I0/R,M*I0/(2*R)],kind='耦合回路积分')
    check('异阻耦合回路总热量',got[4],L*I0*I0/2,kind='耦合回路积分')
    t=.8/ls;yi=rk(fn,[I0,0,0,0,0],t)
    A=(lf-R*L/D)/(lf-ls)*I0
    exact=[A*math.exp(-ls*t)+(I0-A)*math.exp(-lf*t),R*M*I0/(D*(lf-ls))*(math.exp(-ls*t)-math.exp(-lf*t))]
    check('异阻耦合回路双指数',yi[:2],exact,kind='耦合回路积分')

for L1,L2,M in [(9,4,3),(9,1,2),(2,3,.8),(1.5,.9,-.5)]:
    mat=np.array([[L1,M],[M,L2]]);d=np.linalg.solve(mat,np.ones(2))
    Lp=(L1*L2-M*M)/(L1+L2-2*M)
    check('并联端口电感',1/d.sum(),Lp,kind='线性方程求解')
    check('并联支路场能',d@mat@d/2,1/(2*Lp),kind='线性方程求解')
    check('串联相助电感',np.ones(2)@mat@np.ones(2),L1+L2+2*M)
    check('串联相抵电感',np.array([1,-1])@mat@np.array([1,-1]),L1+L2-2*M)
for L in [.7,1.3,2.5]:
    for M in [-.4*L,0,.7*L]:
        mat=np.full((3,3),M,dtype=float);np.fill_diagonal(mat,L)
        check('三个等价线圈本征值',np.sort(np.linalg.eigvalsh(mat)),np.sort([L+2*M,L-M,L-M]),kind='矩阵本征值')

for x in [.2,.6,1.2,2.2]:
    L0=1.3;lam=.8;I0=1.1;psi=L0*I0;m=.9;h=1e-5
    W=lambda x:psi*psi/(2*(L0+lam*x))
    F=- (W(x+h)-W(x-h))/(2*h)
    check('固定磁通链磁力',F,psi*psi*lam/(2*(L0+lam*x)**2),kind='有限差分')
    v=I0*math.sqrt(L0/m*lam*x/(L0+lam*x))
    check('自由部件动能与场能',m*v*v/2+W(x),W(0))
    source=quad(lambda q:np.full_like(q,I0*I0*lam),0,x)
    work=quad(lambda q:np.full_like(q,I0*I0*lam/2),0,x)
    check('恒流可变电感供能',source,2*work)

# 几何闭合回路的Neumann积分，对照轴上偶极近似与距离导数。
def mutual(a,b,d,n=180):
    theta=(np.arange(n)+.5)*2*math.pi/n
    p=np.stack([a*np.cos(theta),a*np.sin(theta),np.zeros(n)],axis=1)
    q=np.stack([b*np.cos(theta),b*np.sin(theta),np.full(n,d)],axis=1)
    dp=np.stack([-a*np.sin(theta),a*np.cos(theta),np.zeros(n)],axis=1)*2*math.pi/n
    dq=np.stack([-b*np.sin(theta),b*np.cos(theta),np.zeros(n)],axis=1)*2*math.pi/n
    distance=np.linalg.norm(p[:,None,:]-q[None,:,:],axis=2)
    return float(np.sum((dp@dq.T)/distance)/(4*math.pi))
for a,b,d in [(1,.8,30),(.6,1.2,25),(.9,.5,40)]:
    M=mutual(a,b,d);C=math.pi*a*a*b*b/2
    check('远场互感与几何积分',M,C/d**3,tol=3*(a*a+b*b)/d**2,kind='几何积分渐近')
    h=d*1e-4;F=(mutual(a,b,d+h)-mutual(a,b,d-h))/(2*h)
    check('远场力与几何积分导数',F,-3*C/d**4,tol=6*(a*a+b*b)/d**2,kind='几何积分渐近')

for alpha,beta,L2,I0,M0 in [(1,1,1.3,.8,.6),(.4,1.2,.8,1.1,.3),(1.5,.3,1.4,.9,.7),(.7,.71,1.1,.6,.4),(1,2,1.2,1.3,.5)]:
    R=beta*L2;T=30/min(alpha,beta)
    def fn(t,y):
        i,Q,A,Ws=y;M=M0*math.exp(-alpha*t)
        di=(alpha*I0*M-R*i)/L2
        return np.array([di,R*i*i,alpha*I0*M*i,I0*M*(di-alpha*i)])
    got=rk(fn,[0,0,0,0],T,n=20000)
    Q=I0*I0*M0*M0/(2*L2)*alpha/(alpha+beta)
    check('移动次级焦耳热',got[1],Q,kind='运动与回路耦合积分')
    check('移动次级机械功',got[2],Q,kind='运动与回路耦合积分')
    check('移动次级电源净功',got[3]+Q,Q,kind='运动与回路耦合积分')
    t=1/max(alpha,beta);yi=rk(fn,[0,0,0,0],t)
    If=alpha*I0*M0/L2*t*math.exp(-alpha*t) if alpha==beta else alpha*I0*M0/L2/(beta-alpha)*(math.exp(-alpha*t)-math.exp(-beta*t))
    check('移动次级双指数或重根解',yi[0],If,kind='运动与回路耦合积分')
    M=M0*math.exp(-alpha*t)
    energy=L2*yi[0]**2/2+M*I0*yi[0]
    check('移动次级有限时能量',yi[3]+yi[2]-yi[1],energy,kind='运动与回路耦合积分')

for L2,R2,M,rate,T in [(1,.8,.4,.7,1.2),(.6,1.5,.3,.4,.9),(2,.5,.8,1.1,1.7)]:
    tau=L2/R2
    def ramp(t,y):return np.array([(-M*rate-R2*y[0])/L2,y[0]])
    mid=rk(ramp,[0,0],T)
    check('给定原电流斜坡次级电流',mid[0],-M*rate/R2*(1-math.exp(-T/tau)),kind='强迫回路积分')
    def tail(t,y):return np.array([-R2*y[0]/L2,y[0]])
    end=rk(tail,mid,25*tau)
    check('原电流斜坡完整传输电荷',end[1],-M*rate*T/R2,kind='强迫回路积分')

for L0,lam,I0,m,T in [(1,.6,.8,1.3,2),(.8,1.1,1.4,.7,1.5),(1.5,.4,.6,1,3)]:
    def fn(t,y):
        x,v,i=y;L=L0+lam*x
        return np.array([v,lam*i*i/(2*m),-lam*v*i/L])
    x,v,i=rk(fn,[0,0,I0],T)
    check('自由可变电感回路磁通链',(L0+lam*x)*i,L0*I0,kind='运动与回路耦合积分')
    ve=I0*math.sqrt(L0/m*lam*x/(L0+lam*x))
    check('自由可变电感回路速度',v,ve,kind='运动与回路耦合积分')
    check('自由可变电感总能量',m*v*v/2+(L0+lam*x)*i*i/2,L0*I0*I0/2,kind='运动与回路耦合积分')
for theta in [.2,.8,1.5,2.5]:
    M0=.7;I1=.8;I2=1.2;h=1e-5
    force=(M0*math.cos(theta+h)-M0*math.cos(theta-h))/(2*h)*I1*I2
    check('转动线圈磁转矩',force,-M0*I1*I2*math.sin(theta),kind='有限差分')
for x in [.3,.8,1.5]:
    M0=.7;d=1.1;I0=.9;L1=1.4;L2=.8;h=1e-5
    def W(x):
        M=M0*math.exp(-x/d);i2=-M*I0/L2
        return (L1*I0*I0+L2*i2*i2)/2+M*I0*i2
    derivative=(W(x+h)-W(x-h))/(2*h)
    M=M0*math.exp(-x/d);F=I0*I0*M*M/(L2*d)
    check('零磁通链次级场能与排斥力',derivative,F,kind='有限差分')

for alpha,L2,I0,M0 in [(1,1.2,.8,.6),(.6,.7,1.1,.4),(1.8,1.4,.6,.7)]:
    def fn(t,y):
        i,A,Ws=y;M=M0*math.exp(-alpha*t);di=alpha*I0*M/L2
        return np.array([di,alpha*I0*M*i,I0*M*(di-alpha*i)])
    end=rk(fn,[0,0,0],30/alpha)
    Ef=I0*I0*M0*M0/(2*L2)
    check('严格无电阻移动次级末电流',end[0],I0*M0/L2,kind='运动与回路耦合积分')
    check('严格无电阻移动次级机械功',end[1],Ef,kind='运动与回路耦合积分')
    check('严格无电阻移动次级电源净功',end[2]+Ef,Ef,kind='运动与回路耦合积分')

mu0=4*math.pi*1e-7;sigma=5.8e7;a=.001
for f,expected in [(50,.00935),(1e6,.0000661)]:
    delta=1/math.sqrt(math.pi*mu0*sigma*f)
    check('趋肤深度书面有效位数',delta,expected,tol=.001,kind='数值代入与有效位数')
check('注入动能书面有效位数',math.sqrt(.5110**2+.2998**2)-.5110,.08145,tol=5e-5,kind='数值代入与有效位数')
check('最终动能书面有效位数',math.sqrt(.5110**2+14.99**2)-.5110,14.49,tol=.0002,kind='数值代入与有效位数')

groups={}
for row in rows:
    group=groups.setdefault(row['kind'],dict(count=0,maximum_normalized_error=0))
    group['count']+=1;group['maximum_normalized_error']=max(group['maximum_normalized_error'],row['normalized_error'])
report=dict(count=len(rows),all_passed=True,description='同一物理关系在不同参数下的实例；包含场与回路积分、有限差分、几何积分、矩阵本征值、缓变和远场近似、书面有效位数核对，不代表独立公式数量。',maximum_normalized_error=max(q['normalized_error'] for q in rows),groups=groups,checks=rows)
(ROOT/'检查/第十三讲_公式核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))
