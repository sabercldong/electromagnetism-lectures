"""独立代入、场积分、边界线性方程、差分及守恒方程求根。
一条记录计一次参数实例的一次数值比较，积分网格点不计次数。
不导入编写脚本，也不读取其数值作为计算输入。
"""
from pathlib import Path
import math,json
import numpy as np
c=299792458.;mu=4*math.pi*1e-7;eps=1/(mu*c*c);eta=mu*c
rows=[]
def check(g,n,x,y,rtol=5e-4,atol=1e-15,method='独立参数代入'):
    a=np.asarray(x);b=np.asarray(y);err=float(np.max(abs(a-b)))
    tol=atol+rtol*float(np.max(abs(b)))
    def value(z):
        if np.iscomplexobj(z):return {'real':z.real.tolist(),'imag':z.imag.tolist()}
        return z.tolist()
    rows.append(dict(group=g,name=n,computed=value(a),reference=value(b),absolute_error=err,tolerance=tol,method=method,passed=err<=tol))
def integ(f,a,b,n=100001):
    x=np.linspace(a,b,n);return np.trapz(f(x),x)
def d(f,x,h=1e-5):return (f(x+h)-f(x-h))/(2*h)
def root(f,a,b):
    fa=f(a)
    assert fa*f(b)<=0
    for _ in range(90):
        m=(a+b)/2;fm=f(m)
        if fa*fm<=0:b=m
        else:a=m;fa=fm
    return (a+b)/2

# W1：矢量叉积与阻抗，不把单分量误当振幅。
n=np.array([2.,1.,2.])/3;E=np.array([30.,0.,-30.]);H=np.cross(n,E)/(eta/2)
check('W1','横向条件',np.dot(n,E),0,atol=1e-13)
check('W1','波长m',c/(2e8),1.499)
check('W1','H幅度A/m',np.linalg.norm(H),.2252)
check('W1','B幅度μT',mu*np.linalg.norm(H)*1e6,.2830)
check('W1','平均能流矢量',np.cross(E,H)/2,4.778*n)

# W2：时间积分协方差，对角化并直接验证椭圆。
ph=np.linspace(0,2*math.pi,100001);xy=np.array([4*np.cos(ph),2*np.cos(ph+math.pi/3)])
C=np.array([[np.trapz(xy[i]*xy[j],ph)/(2*math.pi) for j in range(2)] for i in range(2)])
check('W2','协方差时间积分',C,[[8,2],[2,2]],method='100001点周期积分')
eig,vec=np.linalg.eigh(C)
check('W2','长短半轴V/m',np.sqrt(2*eig[::-1]),[4.149,1.670])
angle=math.atan2(vec[1,-1],vec[0,-1])*180/math.pi%180
check('W2','长轴转角度',angle,16.85)
check('W2','椭圆恒等式',np.max(abs(xy[0]**2/16+xy[1]**2/4-xy[0]*xy[1]/8-.75)),0,atol=3e-15)
check('W2','强度及分析器输出',np.array([np.trace(C),C[0,0],C[1,1]])/eta,[.02655,.02124,.005309])

# W3：时间能流与同一时刻空间能量积分。
tau=20e-9;A=.01;Em=200
U=A*integ(lambda t:Em**2*np.sin(math.pi*t/tau)**4/eta,0,tau)
check('W3','能量时间积分nJ',U*1e9,7.963,method='100001点时间积分')
check('W3','空间能量积分',A*integ(lambda z:eps*Em**2*np.sin(math.pi*z/(c*tau))**4,0,c*tau),U,rtol=1e-10,method='100001点空间积分')
check('W3','脉冲长度m',c*tau,5.996)
check('W3','动量kg m/s',U/c,2.656e-17,atol=1e-21)

# W4：侧面通量、体积热和局域散度。
a=.0005;L=.3;V=6;I=2;Ez=V/L;j=I/(math.pi*a*a)
check('W4','电流密度A/m²',j,2.546e6)
check('W4','侧面磁场A/m',I/(2*math.pi*a),636.6)
Sr=lambda r:-Ez*j*r/2
check('W4','侧面能流W/m²',Sr(a),-1.273e4)
check('W4','侧面总输入功率W',-Sr(a)*2*math.pi*a*L,12)
check('W4','体积分热功率W',integ(lambda r:j*Ez*2*math.pi*r*L,0,a),12,method='100001点柱坐标积分')
for r in [a/4,3*a/4]:
    check('W4','局域散度r='+str(r),d(lambda x:x*Sr(x),r,1e-9)/r,-j*Ez,rtol=1e-7,method='径向中央差分')

# W5：TEM场及电磁储能直接截面积分。
a=.001;b=.004;ep=2.25*eps;ln=math.log(b/a);Z=eta/1.5/(2*math.pi)*ln;V=10;I=V/Z
Er=lambda r:V/(r*ln);Hp=lambda r:I/(2*math.pi*r)
P=integ(lambda r:Er(r)*Hp(r)*math.pi*r,a,b)
UE=integ(lambda r:ep*Er(r)**2/4*2*math.pi*r,a,b)
UH=integ(lambda r:mu*Hp(r)**2/4*2*math.pi*r,a,b)
check('W5','特性阻抗Ω',Z,55.41)
check('W5','功率场积分W',P,.9023,method='100001点截面积分')
check('W5','电磁储能相等',UE,UH,rtol=1e-8,method='100001点截面积分')
check('W5','单位长度储能nJ/m',(UE+UH)*1e9,4.515,method='100001点截面积分')
check('W5','能量速度',P/(UE+UH),c/1.5,rtol=1e-8)

# W6：解两条连续方程后检查三个空间相位的净能流。
eta2=eta/1.5
r,t=np.linalg.solve([[1,-1],[1/eta,1/eta2]],[-1,1/eta])
check('W6','边界方程解',[r,t],[-.2,.8],rtol=1e-10,method='二维线性方程求解')
check('W6','功率率和',r*r+eta/eta2*t*t,1,rtol=1e-10)
check('W6','电場腹节点峰值及驻波比',[10*(1+abs(r)),10*(1-abs(r)),(1+abs(r))/(1-abs(r))],[12,8,1.5])
for p in [-.3,-1.2,-2.7]:
    ee=10*(np.exp(-1j*p)+r*np.exp(1j*p));hh=10/eta*(np.exp(-1j*p)-r*np.exp(1j*p))
    check('W6','净能流相位'+str(p),float(np.real(ee*hh.conjugate())/2),.1274,method='相量场直接叉积')

# W7：无量纲局域能量连续式中央差分。
phase=math.pi/4;z=-math.pi/4
ex=20*math.sin(phase)*math.sin(z);hy=20/eta*math.cos(phase)*math.cos(z)
check('W7','指定点电磁场',[ex,hy],[-10,.02655])
check('W7','指定点能量J/m³',(eps*ex*ex+mu*hy*hy)/2,8.854e-10)
check('W7','指定点能流W/m²',ex*hy,-.2654)
for p,z in [(.4,-.2),(.8,-1.1),(1.4,-2.1)]:
    uu=lambda q:2*(math.sin(q)**2*math.sin(z)**2+math.cos(q)**2*math.cos(z)**2)
    ss=lambda x:math.sin(2*p)*math.sin(2*x)
    check('W7','局域连续式'+str((p,z)),d(uu,p)+d(ss,z),0,atol=8e-10,method='时间与空间中央差分')

# W8：精确传播根、边界反射与内部耗散积分。
w=2*math.pi*1e6;sig=5.8e7;q=np.sqrt(-w*w*mu*eps+1j*w*mu*sig);zs=1j*w*mu/q
Hs=200/(eta+zs);Es=zs*Hs;delta=1/q.real;absorb=.5*zs.real*abs(Hs)**2
check('W8','趋肤深度μm',delta*1e6,66.09)
check('W8','表面电阻mΩ',zs.real*1e3,.2609)
check('W8','吸收W/m²',absorb,3.677e-5)
check('W8','焦耳热深度积分',integ(lambda z:.5*sig*abs(Es)**2*np.exp(-2*q.real*z),0,15*delta),absorb,rtol=1e-7,method='100001点深度积分')
rr=(zs-eta)/(zs+eta)
check('W8','吸收率边界能流',1-abs(rr)**2,2.771e-6,rtol=5e-4)

# W9：入射减反射动量通量，而非再次代入压强公式。
th=math.pi/6;pin=1000*.02*math.cos(th)
ni=np.array([math.sin(th),-math.cos(th)]);nr=np.array([math.sin(th),math.cos(th)])
F=pin/c*(ni-.8*nr)
check('W9','吸收功率W',.2*pin,3.464)
check('W9','法切光力N',[-F[1],F[0]],[9.006e-8,5.778e-9],method='出入动量通量相减')

# W10：TE10截面上的三个场分量直接积分。
a=.03;b=.01;w=2*math.pi*7.5e9;kc=math.pi/a;beta=math.sqrt((w/c)**2-kc*kc);e0=100
ey=lambda x:e0*np.sin(kc*x);hx=lambda x:-beta/(w*mu)*ey(x);hz=lambda x:kc/(w*mu)*e0*np.cos(kc*x)
power=b*integ(lambda x:-ey(x)*hx(x)/2,0,a)
energy=b*integ(lambda x:(eps*ey(x)**2+mu*(hx(x)**2+hz(x)**2))/4,0,a)
check('W10','截止GHz',c/(2*a)/1e9,4.997)
check('W10','导波波长mm',2*math.pi/beta*1e3,53.60)
check('W10','功率mW',power*1e3,1.485,method='100001点截面积分')
check('W10','储能pJ/m',energy*1e12,6.641,method='100001点截面积分')
check('W10','能量速度/c',power/energy/c,.7458)
betaf=lambda om:math.sqrt((om/c)**2-kc*kc)
check('W10','色散导数群速度',1/d(betaf,w,w*1e-5),power/energy,rtol=1e-8,method='色散中央差分与能流比')

# W11：电流积分及辐射方向图球面积分。
ell=.1;w=2*math.pi*1e8;ic=.4;r=100
leff=integ(lambda z:1-2*np.abs(z)/ell,-ell/2,ell/2)
ef=mu*w*ic*leff/(4*math.pi*r)
pr=2*math.pi*r*r*integ(lambda th:ef*ef*np.sin(th)**3/(2*eta),0,math.pi)
rrad=2*pr/ic**2
check('W11','有效长度m',leff,.05,method='100001点电流积分')
check('W11','辐射电阻Ω',rrad,.2195,method='100001点辐射球面积分')
check('W11','辐射功率mW',pr*1e3,17.56)
check('W11','效率',rrad/(rrad+.5),.3051)
check('W11','赤道远场V/m',ef,.01257)
check('W11','方向性系数',4*math.pi*r*r*ef**2/(2*eta)/pr,1.5,rtol=1e-8)

# W12：独立积分电流平方与二阶ODE残差。
L=1e-7;C=1e-11;R=5;gam=R/(2*L);w0=1/math.sqrt(L*C);wd=math.sqrt(w0*w0-gam*gam);q0=C*1000
qq=lambda t:q0*np.exp(-gam*t)*(np.cos(wd*t)+gam/wd*np.sin(wd*t))
qd=lambda t:-q0*w0*w0/wd*np.exp(-gam*t)*np.sin(wd*t)
rad=2*integ(lambda t:qd(t)**2,0,20/gam,300001)
loss=3*integ(lambda t:qd(t)**2,0,20/gam,300001)
check('W12','阻尼频率rad/s',wd,9.997e8)
check('W12','振幅时间ns',1/gam*1e9,40)
check('W12','辐射热分配μJ',np.array([rad,loss])*1e6,[2,3],rtol=1e-8,method='300001点时间积分')
check('W12','平均辐射mW',rad*50*1e3,.1)
for t in [1e-9,5e-9,2e-8]:
    check('W12','回路ODE残差'+str(t),(L*d(qd,t,1e-14)+R*qd(t)+qq(t)/C)/1000,0,atol=2e-10,method='电流中央差分')

# T1：对两个不同轮廓差分验证完整旋度关系。
F=lambda u:np.sin(2*u);G=lambda u:.3*np.cos(3*u)
for t,z in [(.2,.3),(.7,-.4),(1.1,.6)]:
    EE=lambda zz:F(t-zz)+G(t+zz);BB=lambda tt:F(tt-z)-G(tt+z)
    HH=lambda zz:F(t-zz)-G(t+zz);ET=lambda tt:F(tt-z)+G(tt+z)
    check('T1','法拉第式'+str((t,z)),-d(EE,z),d(BB,t),rtol=1e-8,method='无量纲中央差分')
    check('T1','安培式'+str((t,z)),-d(HH,z),d(ET,t),rtol=1e-8,method='无量纲中央差分')
check('T1','候选纵向场散度',d(lambda z:math.cos(.8-z),.3),math.sin(.5),method='中央差分')

# T2：任意分析器角的周期积分。
check('T2','圆偏振强度',9/eta,.02389)
for th in [0,.3,1.1]:
    mean=integ(lambda ph:9*np.cos(ph-th)**2/eta,0,2*math.pi)/(2*math.pi)
    check('T2','分析器强度角'+str(th),mean,.01195,method='100001点时间积分')

# T3：面电流跳变和做功核对。
K=1;Es=-eta*K/2;hl=Es/eta;hr=-Es/eta;tau=1e-8
check('T3','H跳变A/m',hr-hl,K,rtol=1e-10)
check('T3','电源功率等于两侧发射',-K*Es,abs(-Es*hr)+abs(-Es*hl),rtol=1e-10)
check('T3','发射总能量μJ/m²',-K*Es*tau*1e6,1.884)
check('T3','波包厚度m',c*tau,2.998)

# T4：直接周期平均E×H和两项储能。
ph=np.linspace(0,2*math.pi,100001);ee=np.outer(2*np.cos(ph),[0,1,0]);hh=np.outer(np.cos(ph)/eta,[-1,0,1])
sb=np.trapz(np.cross(ee,hh),ph,axis=0)/(2*math.pi)
ub=np.trapz((eps*np.sum(ee*ee,axis=1)+mu*np.sum(hh*hh,axis=1))/2,ph)/(2*math.pi)
check('T4','矢量能流周期积分',sb,np.array([1,0,1])/eta,rtol=1e-10,method='100001点周期积分')
check('T4','能流与储能比',np.linalg.norm(sb)/(c*ub),2*math.sqrt(2)/3,rtol=1e-10,method='100001点周期积分')

# T5：四个切向边界条件解，独立于几何级数公式。
nc=1.5;nf=2.25;df=100e-9
for lam,target in [(600e-9,0),(570e-9,1.183e-3)]:
    ph=2*math.pi*nc*df/lam;ec=eta/nc;ef=eta/nf
    # 未知量r,A,B,t，膜内前后向峰值均按固定x轴记号。
    mat=np.array([[1,-1,-1,0],[-1/eta,-1/ec,1/ec,0],
        [0,np.exp(-1j*ph),np.exp(1j*ph),-1],
        [0,np.exp(-1j*ph)/ec,-np.exp(1j*ph)/ec,-1/ef]],complex)
    rr,aa,bb,tt=np.linalg.solve(mat,[-1,-1/eta,0,0])
    check('T5','反射率λ='+str(lam),abs(rr)**2,target,atol=1e-14,method='四界面方程求解')
    check('T5','能量守恒λ='+str(lam),abs(rr)**2+nf*abs(tt)**2,1,rtol=1e-10)

# T6：复数传播根、深度热积分及近似误差。
w=2*math.pi*1e7;ep=4*eps;sig=w*ep;q=np.sqrt(-w*w*mu*ep+1j*w*mu*sig);zz=1j*w*mu/q
check('T6','传播系数1/m',[q.real,q.imag],[.1908,.4605])
check('T6','复阻抗实虚Ω',[zz.real,zz.imag],[146.3,60.62])
pj=integ(lambda z:sig/2*np.exp(-2*q.real*z),0,15/q.real)
check('T6','热积分W/m²',pj,2.916e-3,method='100001点深度积分')
check('T6','边界功率等于热积分',.5*np.real(1/zz),pj,rtol=1e-7)
ag=math.sqrt(w*mu*sig/2)
check('T6','良导体近似相对误差',[ag/q.real-1,ag/q.imag-1],[.554,-.356],rtol=.002)

# T7：两个偏振的边界方程分别解，并求p零点。
def fresnel(th):
    tt=math.asin(math.sin(th)/1.5);ci=math.cos(th);ct=math.cos(tt)
    rs,ts=np.linalg.solve([[1,-1],[ci/eta,ct/(eta/1.5)]],[-1,ci/eta])
    rp,tp=np.linalg.solve([[ci,-ct],[1/eta,1/(eta/1.5)]],[-ci,1/eta])
    return rs,rp,ts,tp,tt
th=root(lambda x:fresnel(x)[1],.1,1.4);rs,rp,ts,tp,tt=fresnel(th)
check('T7','p零点入射角度',th*180/math.pi,56.31,method='连续方程求解与二分求根')
check('T7','s反射率',rs*rs,25/169,rtol=1e-10)
check('T7','透反射方向直角',th+tt,math.pi/2,rtol=1e-10)
check('T7','s与p能量率',[rs*rs+1.5*math.cos(tt)/math.cos(th)*ts*ts,rp*rp+1.5*math.cos(tt)/math.cos(th)*tp*tp],[1,1],rtol=1e-10)

# T8：复波矢代入旋度关系与复Poynting。
k0=2*math.pi/600e-9;kx=k0*1.5*math.sin(math.pi/3);kap=math.sqrt(kx*kx-k0*k0);w=c*k0
ev=np.array([0,1,0],complex);hv=np.cross(np.array([kx,0,-1j*kap]),ev)/(w*mu);sc=np.cross(ev,hv.conjugate())/2
check('T8','衰减系数1/m',kap,8.683e6)
check('T8','幅度穿透nm',1/kap*1e9,115.2)
check('T8','法向平均功率',sc[2].real,0,atol=1e-15,method='复波矢叉积')
check('T8','切向功率与虚部',[sc[0].real,sc[2].imag],[kx/(2*w*mu),kap/(2*w*mu)],rtol=1e-10)

# T9：损耗独立表面积分及传播根。
w=2*math.pi*1e8;a=.001;b=.004;sig=5.8e7;Rs=math.sqrt(w*mu/(2*sig));Z=eta/1.5/(2*math.pi)*math.log(4)
surface=Rs/2*((1/(2*math.pi*a))**2*2*math.pi*a+(1/(2*math.pi*b))**2*2*math.pi*b)
rp=2*surface;Lp=mu/(2*math.pi)*math.log(4);Cp=2*math.pi*(2.25*eps)/math.log(4)
qp=np.sqrt((rp+1j*w*Lp)*1j*w*Cp);alpha=rp/(2*Z)
check('T9','交流电阻Ω/m',rp,.5190,method='两导体表面损耗积分')
check('T9','衰减1/m',alpha,.004683)
check('T9','弱损耗近似与精确传播根',qp.real,alpha,rtol=3e-6)
check('T9','20m功率比',math.exp(-2*alpha*20),.8292)
check('T9','弱损耗比',rp/(w*Lp),.00298,rtol=.002)

# T10：截止以下复波阻抗无实部功率。
a=.03;w=2*math.pi*3.5e9;kap=math.sqrt((math.pi/a)**2-(w/c)**2);hx=1j*kap/(w*mu)
check('T10','衰减1/m',kap,74.73)
check('T10','20mm幅比',math.exp(-.02*kap),.2243)
check('T10','平均轴功率',-.5*np.real(hx.conjugate()),0,atol=1e-15)

# T11：频率导数数值计算，保留介质储能。
w0=2*math.pi*1e9;nfun=lambda w:1.5+.1*(w-w0)/w0;kfun=lambda w:w*nfun(w)/c
vg=1/d(kfun,w0,w0*1e-5);der=d(lambda w:w*eps*nfun(w)**2,w0,w0*1e-5)
ubar=der/4+mu*(1.5/eta)**2/4;sb=1.5/(2*eta)
check('T11','群速度/c',vg/c,.625,rtol=1e-8,method='频率中央差分')
check('T11','色散储能系数',der/eps,2.55,rtol=1e-8,method='频率中央差分')
check('T11','能量速度/c',sb/ubar/c,.625,rtol=1e-8)
delta=.01*w0;pherr=(kfun(w0+delta)-kfun(w0)-d(kfun,w0,w0*1e-5)*delta)*20
check('T11','二阶传播相位rad',pherr,.004192,method='完整色散式减线性传播相位')

# T12：相干角及相对论阈值。
n=1.33;beta=.9
check('T12','切伦科夫角度',math.acos(1/(beta*n))*180/math.pi,33.34)
check('T12','阈值动能MeV',(.511/math.sqrt(1-1/n**2)-.511),.264,rtol=.002)
check('T12','相位匹配残差',1-n*beta*math.cos(math.acos(1/(n*beta))),0,atol=1e-15)

# T13：磁节点反求波长和频率误差端点。
lam=2*(2.25-.75);f=c/lam;em=2*math.pi*f*math.pi*.01**2*20/c
check('T13','频率MHz',f/1e6,99.93)
check('T13','误差端点MHz',[c/(lam+.02)/1e6,c/(lam-.02)/1e6],[99.27,100.60])
check('T13','接收峰值mV',em*1e3,13.16)
check('T13','两磁节点',[math.cos(2*math.pi*.75/lam),math.cos(2*math.pi*2.25/lam)],[0,0],atol=1e-14)

# T14：含正确球面积元的数值积分及小锥极限。
norm=integ(lambda th:np.sin(th)**3,0,math.pi)
F60=integ(lambda th:np.sin(th)**3,0,math.pi/3)/norm
check('T14','单锥积分',F60,5/32,rtol=1e-8,method='100001点立体角积分')
check('T14','赤道带积分',integ(lambda th:np.sin(th)**3,math.pi/3,2*math.pi/3)/norm,11/16,rtol=1e-8,method='100001点立体角积分')
check('T14','轴向小锥渐近',integ(lambda th:np.sin(th)**3,0,.01)/norm/(.01**4),3/16,rtol=1e-4,method='小角立体角积分')

# T15：负载功率求导根与完整串联半功率根差。
w=2*math.pi*1e8;L=1e-6;A=.001;R0=10;em=w*A/c;C=1/(w*w*L)
power=lambda rl:em*em*rl/(2*(R0+rl)**2)
opt=root(lambda rl:d(power,rl,1e-4),1,30)
check('T15','最优负载Ω',opt,10,rtol=1e-8,method='功率差分导数二分求根')
check('T15','谐振电容pF',C*1e12,2.533)
check('T15','最大功率nW',power(opt)*1e9,54.91)
R=R0+opt;lower=root(lambda x:x*L-1/(x*C)+R,w/2,w);upper=root(lambda x:x*L-1/(x*C)-R,w,2*w)
check('T15','半功率全带宽MHz',(upper-lower)/(2*math.pi*1e6),3.183,method='两侧半功率方程分别求根')

# T16：直接叠加两个延迟相量；球积分检查交叉项不为零。
for phi,target in [(0,0),(math.pi/2,4),(math.pi,0),(math.pi/3,2)]:
    factor=abs(1+np.exp(1j*math.pi*math.cos(phi)))**2
    check('T16','方向因子φ='+str(phi),factor,target,atol=1e-14,method='延迟场相量直接相加')
th=np.linspace(0,math.pi,801);ph=np.linspace(0,2*math.pi,1601)
arrayfactor=2+2*np.cos(math.pi*np.sin(th[:,None])*np.cos(ph[None,:]))
combined=np.trapz(np.trapz(arrayfactor*np.sin(th[:,None])**3,ph,axis=1),th)
single=2*math.pi*integ(lambda th:np.sin(th)**3,0,math.pi)
# x向间距、z向偶极的球面交叉积分= -3/(2π²) 单源功率。
check('T16','三维总辐射与互辐射',combined/single,2-3/(math.pi**2),rtol=1e-8,method='801×1601点球面角积分')

# T17：逐个解E连续和H跳变方程，含多个相位实例。
def sheet(g,a,d):
    # b-c=d-a；(a-b-c+d)=g(a+b)
    b,c=np.linalg.solve([[1,-1],[1+g,1]], [d-a,(1-g)*a+d])
    return b,c,a+b
for g in [.1,2,8]:
    bout,cout,es=sheet(g,1,0);A1=1-abs(bout)**2-abs(cout)**2
    check('T17','单侧吸收g='+str(g),A1,4*g/(2+g)**2,rtol=1e-10,method='界面方程求解')
    for delta in [0,math.pi/3,math.pi]:
        bout,cout,es=sheet(g,1,np.exp(1j*delta));AA=(2-abs(bout)**2-abs(cout)**2)/2
        check('T17','双侧吸收'+str((g,delta)),AA,4*g/(2+g)**2*(1+math.cos(delta)),rtol=1e-10,atol=2e-15,method='界面方程求解')
        check('T17','焦耳热能流'+str((g,delta)),AA,g*abs(es)**2/2,rtol=1e-10,atol=2e-15,method='界面通量与面耗散分别计算')
G=2/eta;df=G/4e4;delta=math.sqrt(2/(2*math.pi*1e8*mu*4e4))
check('T17','最优膜厚nm',df*1e9,132.7)
check('T17','趋肤深度μm',delta*1e6,251.6)
check('T17','厚度趋肤比',df/delta,5.28e-4,rtol=.002)

# T18：不使用给出的镜速公式，直接解最终能量/动量方程。
for u in [.001,.1,1]:
    # 取静能量Mc²=1，镜动量变量p=PM/(Mc)。Ur=p-u。
    p=root(lambda p:math.sqrt(1+p*p)+(p-u)-(1+u),u,2*u)
    gm=math.sqrt(1+p*p);bf=p/gm;ur=p-u
    check('T18','全脉冲反射能量u='+str(u),ur/u,1/(1+2*u),rtol=1e-9,method='能量动量守恒二分求根')
    check('T18','镜速u='+str(u),bf,((1+2*u)**2-1)/((1+2*u)**2+1),rtol=1e-9,method='守恒求根后计算')
    check('T18','波包频移积分u='+str(u),integ(lambda w:(1+2*w)**(-2),0,u),ur,rtol=1e-9,method='100001点累计入射能量积分')
check('T18','u=.1静镜冲量相对实际误差',(2*.1-11/60)/(11/60),1/11,rtol=1e-10)

# 理论：偶极延迟势空间、时间差分，独立核对全场含远区负号。
# 用无量纲c=1、4πε=1的函数，消去单位系数，保留同一Maxwell关系。
def potentials(rv,t):
    r=np.linalg.norm(rv);n=rv/r;tr=t-r
    pv=np.array([0,0,math.cos(tr)]);pd=np.array([0,0,-math.sin(tr)])
    return np.dot(n,pv)/r**2+np.dot(n,pd)/r,pd/r
def field_formula(rv,t):
    r=np.linalg.norm(rv);n=rv/r;tr=t-r
    pv=np.array([0,0,math.cos(tr)]);pd=np.array([0,0,-math.sin(tr)]);pdd=-pv
    E=(3*n*np.dot(n,pv)-pv)/r**3+(3*n*np.dot(n,pd)-pd)/r**2+np.cross(n,np.cross(n,pdd))/r
    B=np.cross(pd,n)/r**2+np.cross(pdd,n)/r
    return E,B
for rv,t in [(np.array([1.,.5,2.]),.7),(np.array([10.,3.,4.]),1.1),(np.array([80.,10.,0.]),.3)]:
    h=1e-4;grad=[];JA=np.zeros((3,3))
    for j in range(3):
        dv=np.eye(3)[j]*h;pa,aa=potentials(rv+dv,t);pb,ab=potentials(rv-dv,t)
        grad.append((pa-pb)/(2*h));JA[:,j]=(aa-ab)/(2*h)
    Adot=(potentials(rv,t+h)[1]-potentials(rv,t-h)[1])/(2*h)
    Ep=-np.array(grad)-Adot;Bp=np.array([JA[2,1]-JA[1,2],JA[0,2]-JA[2,0],JA[1,0]-JA[0,1]])
    Ef,Bf=field_formula(rv,t)
    check('理论','偶极电场从延迟势求导'+str(rv),Ep,Ef,rtol=1e-7,atol=1e-10,method='延迟势空间与时间中央差分')
    check('理论','偶极磁场从势求旋度'+str(rv),Bp,Bf,rtol=1e-7,atol=1e-10,method='延迟矢势空间中央差分')

groups=sorted({r['group'] for r in rows});expected={f'W{i}' for i in range(1,13)}|{f'T{i}' for i in range(1,19)}|{'理论'}
assert set(groups)==expected
output=dict(count=len(rows),count_description='一项=一个参数实例的一次数值比较；覆盖12道例题、18道习题及偶极理论；积分网格点不计数，次数不等于独立公式数量。',all_passed=all(r['passed'] for r in rows),groups=groups,records=rows,constants={'c':c,'mu0':mu,'epsilon0':eps,'eta0':eta})
path=Path(__file__).resolve().parent/'检查/第二十七讲_公式核验.json'
path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
print(json.dumps({'count':len(rows),'all_passed':output['all_passed'],'failed':[r for r in rows if not r['passed']]},ensure_ascii=False))
assert output['all_passed']
