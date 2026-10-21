"""独立计算记录；一条记录是一次参数实例中的一项数值比较，不是独立公式数。"""
from pathlib import Path
import json,math,re
import numpy as np
mu=4*math.pi*1e-7;ke=8.98755e9;g=9.81;c=299792458.
trap=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
records=[];values={}
def check(group,name,computed,reference,rtol=2e-5,atol=1e-12,method='独立数值代入'):
    z=np.asarray(computed);w=np.asarray(reference)
    err=float(np.max(np.abs(z-w)))
    tol=atol+rtol*float(np.max(np.abs(w)))
    records.append(dict(group=group,name=name,computed=z.tolist(),reference=w.tolist(),
                        absolute_error=err,tolerance=tol,method=method,passed=err<=tol))
def diff(f,x,h):
    return (f(x+h)-f(x-h))/(2*h)
def jac(f,x):
    x=np.array(x,float);cols=[]
    for j in range(len(x)):
        h=max(abs(x[j])*1e-5,1e-9);a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
        cols.append((np.atleast_1d(f(a))-np.atleast_1d(f(b)))/(2*h))
    return np.array(cols).T
def fit(xs):
    js=np.arange(len(xs));X=np.column_stack((np.ones(len(js)),js))
    return np.linalg.lstsq(X,np.array(xs),rcond=None)[0],X

# W1：受力关系反代、有限差分误差传播及三角函数展开。
theta=math.radians(10);ut=math.radians(.2)
fq=lambda a:2*a[0]*math.sin(a[2])*math.sqrt(a[1]*g*math.tan(a[2])/ke)
x=[.2,.0002,theta];q=fq(x)
check('W1','电荷nC',q*1e9,13.62757085,rtol=1e-6)
force=ke*q*q/(2*.2*math.sin(theta))**2
check('W1','水平受力平衡',force,.0002*g*math.tan(theta))
uq=np.sqrt((jac(fq,x)**2)@np.array([.001,.000002,ut])**2).item()
coef=1/math.tan(theta)+1/(2*math.sin(theta)*math.cos(theta))
check('W1','有限差分标准不确定度',uq/q,math.sqrt(.005**2+.005**2+(coef*ut)**2),method='三参数中央差分')
for deg in [2,5,10]:
    t=math.radians(deg);ratio=t/math.sin(t)*math.sqrt(t/math.tan(t))
    check('W1','小角四阶展开_'+str(deg),ratio,1-t**4/30,rtol=2e-6)

# W2：解析解对常系数电路方程和加载前后比较。
C=1.2e-9;R=1/(1/2e9+1/1e7);tau=R*C;V0=200e-12*5/C
check('W2','时间常数ms',tau*1e3,11.94029851)
check('W2','6ms电压',V0*math.exp(-.006/tau),.504180189)
for t in [0,.006,.02]:
    V=lambda s:V0*math.exp(-s/tau)
    check('W2','微分方程_'+str(t),C*diff(V,t,1e-7)+V(t)/R,0,atol=2e-15,method='中央差分方程残差')
check('W2','接表前后时间比',2/tau,167.5)

alpha=mu/(2*math.pi*.02);BH=2*alpha*.6/(.34+.26);Bx=BH*.04
check('W3','水平场微特斯拉',BH*1e6,20)
check('W3','固定横向场微特斯拉',Bx*1e6,.8)
check('W3','正负读数反代',[(alpha*.6+Bx)/BH,(-alpha*.6+Bx)/BH],[.34,-.26])
check('W3','单向反演相对偏差',alpha*.6/.34/BH-1,-.1176470588)

N=400;I=.2;S=4e-4;gap=.002
B=mu*N*I/gap;Lf=lambda gg:mu*N*N*S/gg
F=B*B*S/(2*mu)
check('W4','磁场',B,.05026548246)
check('W4','自感',Lf(gap),.04021238597)
check('W4','力',F,.4021238597)
check('W4','余能位移导数',-.5*I**2*diff(Lf,gap,1e-8),F,method='气隙长度中央差分')
dl=Lf(gap-1e-7)-Lf(gap)
check('W4','微位移电源能量收支',I*I*dl,.5*I*I*dl+.5*I*I*dl)

H=lambda a:sum(mu*130*a*a/(2*(a*a+(z-a/2)**2)**1.5) for z in [0,a])
Bh=H(.15);em=400/(Bh**2*.06**2)
check('W5','双线圈轴场求和',Bh,(4/5)**1.5*mu*130/.15,method='单线圈轴场叠加')
check('W5','荷质比',em,1.82963e11)
ef=lambda p:2*p[0]/(((4/5)**1.5*mu*130*p[1]/p[2])**2*p[3]**2)
ue=np.sqrt((jac(ef,[200,1,.15,.06])**2)@np.array([1,.005,.001,.0005])**2).item()
check('W5','荷质比误差传播',ue/em,math.sqrt((1/200)**2+(2*.005)**2+(2*.001/.15)**2+(2*.0005/.06)**2),method='四参数中央差分')
rel=400/(Bh**2*.06**2-200**2/c**2)
check('W5','相对论修正百分比',(rel/em-1)*100,.0204,rtol=.0025)

rs=np.linspace(0,.08,10001);E=float(trap(.5*100*rs,rs));Ig=E/.2
torque=float(trap(Ig*.5*rs,rs))
check('W6','动生电动势径向积分',E,.16,method='10001点梯形积分')
check('W6','转矩径向积分',torque,.00128,method='10001点梯形积分')
check('W6','机械与焦耳功率',100*torque,Ig**2*.2)

Q=-.012/40
check('W7','磁通链变化',-50*Q,.015)
check('W7','每匝磁通',-50*Q/200,75e-6)
check('W7','自感端点修正',-50*Q-.5*(-.001),.0155)

ph=.002;a=.02;v=.3;R=.1
G=lambda x:-ph*x/a**2*np.exp(-x*x/(2*a*a))
xx=np.linspace(-8*a,8*a,50001);heat=float(trap(G(xx)**2*v/R,xx))
check('W8','完整热量积分',heat,v*ph**2*math.sqrt(math.pi)/(2*R*a),method='50001点空间积分')
check('W8','最大阻力',G(a)**2*v/R,.01103638324)
for x in [-a,0,a]:
    fx=-G(x)**2*v/R;i=-G(x)*v/R
    check('W8','局部机械热功率_'+str(x),-fx*v,R*i*i)
check('W8','低自感时间尺度比',(.0001/.1)/(.02/.3),.015)

om=2*math.pi*50;lr=2e-6;rr=.0002;M=20e-6;Mp=-M/.05;I1=5
I2=-1j*om*M*I1/(rr+1j*om*lr)
Fjump=Mp*(I1*I2.conjugate()).real;Pjump=rr*abs(I2)**2
check('W9','相量实部平均力',Fjump,.090800033164,method='复数电路求解')
check('W9','焦耳功率',Pjump,.454000165821)
tt=np.linspace(0,2*math.pi,20001)
itime=np.sqrt(2)*np.real(I2*np.exp(1j*tt))
ftime=Mp*np.sqrt(2)*I1*np.cos(tt)*itime
check('W9','一周期力平均',float(trap(ftime,tt)/(2*math.pi)),Fjump,method='20001点时域积分')
values['W9_zeq_mm']=.05/2*math.log(Fjump/(.008*g))*1000
check('W9','竖直平衡高度mm',values['W9_zeq_mm'],3.65,rtol=.0015)

chi=9e-6;rho=1000;B0=16;b=-mu*rho*g/(chi*B0)
Ub=lambda q:rho*g*q[2]+chi/(2*mu)*((b*q[0]/2)**2+(b*q[1]/2)**2+(B0+b*q[2])**2)
for axis in range(3):
    h=1e-5;xp=np.zeros(3);xm=np.zeros(3);xp[axis]=h;xm[axis]=-h
    curvature=(Ub(xp)+Ub(xm)-2*Ub(np.zeros(3)))/h**2
    reference=chi*b*b/mu*(1 if axis==2 else .25)
    check('W10','能量二阶差分轴_'+str(axis),curvature,reference,rtol=3e-4,method='三维Hessian对角中央差分')
check('W10','重力与磁力',chi*B0*b/mu+rho*g,0,atol=1e-10)
check('W10','竖直频率Hz',math.sqrt(chi*b*b/(mu*rho))/(2*math.pi),1.15306210)

fitw,X=fit([.050,.101,.149,.201,.250])
check('W11','最小二乘截距斜率',fitw,[.0502,.05],method='numpy矩阵最小二乘')
check('W11','读数残差mm',(np.array([.050,.101,.149,.201,.250])-X@fitw)*1000,[-.2,.8,-1.2,.8,-.2])
cov=np.linalg.inv(X.T@X)*.001**2
check('W11','斜率标准误差',math.sqrt(cov[1,1]),.001/math.sqrt(10),method='设计矩阵协方差')
check('W12','偏振背景与泄漏三角验证',.02+1.2*(.5+.01*.5),.626)
check('W12','泄漏功率系数',(.032-.020)/(1.220-.020),.01)

HH=lambda d:math.sin(math.radians(d))**2*math.tan(math.radians(d))
tq=-60/math.log(HH(6)/HH(12))
check('T1','漏电时间s',tq,28.85220750)
check('T1','精确角度模型反代',math.sqrt(HH(6)/HH(12)),math.exp(-30/tq))
check('T2','离子风主项与气流修正',[50e-6*.1/2e-4,50e-6*.1/(2e-4+1/(20000/.1))],[.025,.02439024390])
check('T2','机械电功率效率',.025*1/(20000*50e-6),.025)
f=20;C=.5e-9;R=1e8;xp=2*math.pi*f*R*C;resp=1j*xp/(1+1j*xp)
check('T3','复数频率幅值与相位',[2*abs(resp),math.degrees(np.angle(resp))],[1.975140984,9.04306108],method='复数电路求解')
check('T3','百分之一平台阈值频率',.99/math.sqrt(1-.99**2)/(2*math.pi*R*C),22.33839,rtol=1e-4)

for d in [.02,.04,.007]:
    ss=np.linspace(-.05,.05,20001)
    integral=float(trap(d/(ss*ss+d*d)**1.5,ss))*mu/(4*math.pi)
    theory=mu/(2*math.pi*d)*.05/math.sqrt(.05**2+d*d)
    check('T4','有限导线积分_'+str(d),integral,theory,rtol=1e-7,method='20001点毕奥萨伐尔积分')
check('T4','反演场相对偏差',[math.sqrt(1+t*t)-1 for t in [.4,.8]],[.07703296143,.28062484749])
Kout=.5*.002*.5**2+(-.002)-(-.010)-.001
check('T5','能量与出射速度',[Kout,math.sqrt(2*Kout/.003)],[.00725,2.198484326])
volume=4*math.pi*.001**3/3;chi=9;ce=3*chi/(3+chi);m=volume*ce*.1/mu
check('T6','退磁线性方程',ce*.1/mu,chi*(.1/mu-ce*.1/mu/3))
check('T6','磁矩与饱和力',[m,volume*1e5],[.00075,.0004188790205])
fU=lambda z:-ce*volume*(.1+z)**2/(2*mu)
check('T6','诱导能量梯度力',-diff(fU,0,1e-7),m,method='诱导能量中央差分')
check('T7','两次历史的电流',[(5e-6-10e-6)/10e-6,-10e-6/10e-6],[-.5,-1])
for old in [0,5e-6]:
    ph=lambda z:10e-6+1e-5*z
    U=lambda z:(old-ph(z))**2/(2*10e-6)
    check('T7','历史势能与力_'+str(old),-diff(U,0,1e-5),(old-ph(0))/10e-6*1e-5,method='势能中央差分')
check('T8','恒压升温吸力',1/(1+.004*50)**2,.6944444444)
check('T8','吸力反演温升',(math.sqrt(1/.8)-1)/.004,29.50849719)
K=.5*.5*.08**2;beta=1e-5;R=.2;w=K/(K*K+R*beta);I=(1-K*w)/R
check('T9','电动机稳态转矩',K*I,beta*w)
check('T9','电机转速电流',[w,I],[350.87719298,2.192982456])
check('T9','电机能量',I,R*I*I+beta*w*w)
ell=.04;d=.01;r=(ell*ell+d*d)/(2*d)
check('T10','圆弧几何',ell**2+(d-r)**2,r*r)
check('T10','抛物线反演偏差',(r/(ell**2/(2*d)))**2-1,.12890625)

ph=.001;R=10;T=.02;tau=.005;L=tau*R
icur=lambda t:-ph/(R*(T-tau))*(math.exp(-t/T)-math.exp(-t/tau))
for t in [.001,.01,.03]:
    check('T11','脉冲方程残差_'+str(t),L*diff(icur,t,1e-8)+R*icur(t),-ph/T*math.exp(-t/T),rtol=1e-7,method='中央差分方程残差')
peak=tau*T/(T-tau)*math.log(T/tau)
check('T11','峰值时刻与电流',[peak,icur(peak)],[.009241962407,-.003149802625])
ts=np.linspace(0,.5,100001)
check('T11','完整电荷积分',float(trap([-ph/(R*(T-tau))*(math.exp(-t/T)-math.exp(-t/tau)) for t in ts],ts)),-ph/R,rtol=1e-7,method='100001点时间积分')
for t in [.01,.02,.05]:
    limit=-ph*t/(R*T*T)*math.exp(-t/T)
    te=T*(1+1e-5)
    general=-ph/(R*(T-te))*(math.exp(-t/T)-math.exp(-t/te))
    check('T11','重根极限_'+str(t),general,limit,rtol=1e-5)

G=.05;L=.005;om=20;A=.01
for R in [.02,.1,.5]:
    vv=1j*om*A;ii=-G*vv/(R+1j*om*L)
    tt=np.linspace(0,2*math.pi,20001)
    it=np.real(ii*np.exp(1j*tt));vt=np.real(vv*np.exp(1j*tt))
    P=float(trap(R*it*it,tt)/(2*math.pi))
    gamma=G*G*R/(R*R+om*om*L*L)
    check('T12','耗散周期积分_'+str(R),P,gamma*(om*A)**2/2,method='复数解与20001点平均')
    check('T12','机械与焦耳平均_'+str(R),float(trap(-G*it*vt,tt)/(2*math.pi)),P,method='20001点机械功率积分')
check('T12','最优电阻与最大功率',[om*L,G*G/(2*om*L)*(om*A)**2/2],[.1,.00025])
check('T13','冷热力功率比',[2/1.25,(.5/1.25)/(1/2)],[1.6,.8])
check('T14','近远场电压与功率主项',[2**-3,2**-6,2**-1,2**-2],[.125,.015625,.5,.25])

js=np.arange(6);X=np.column_stack((np.ones(6),js));Ccov=np.linalg.inv(X.T@X)*.0004**2
urel=math.sqrt(Ccov[1,1]/.05**2+.002**2)
check('T15','斜率协方差',math.sqrt(Ccov[1,1]),.0004/math.sqrt(17.5),method='设计矩阵协方差')
check('T15','刻度综合相对误差',urel,.00276715)
check('T15','频率标准误差MHz',c/.1*urel/1e6,8.2957198586)
B0=1.5;gg=.02;gamma=42.577e6
check('T16','共振频率层厚',[gamma*B0,2000/(gamma*gg)],[63865500.,.002348686])
check('T16','能级差',[6.62607015e-34*gamma*B0,6.62607015e-34*gamma*B0/1.602176634e-19],[4.23177283165e-26,2.641264853e-7],atol=0)

m=.04;k=1.6;L=.0001;R=.02;lo=.1;lambdar=.35;lambdac=.725
b0=2*m*lo;G=math.sqrt(2*m*R*(lambdar-lo));Rc=R*(lambdar-lo)/(lambdac-lo)
check('T17','标定参数',[b0,G,Rc],[.008,.02,.008])
fc=lambda a:np.array([math.sqrt(2*m*a[0]*(a[2]-a[1])),
                       a[0]*(a[2]-a[1])/(a[3]-a[1])])
J=jac(fc,[R,lo,lambdar,lambdac]);ugrc=np.sqrt(np.diag(J@np.diag(np.array([.0002,.003,.005,.010])**2)@J.T))
check('T17','相关误差有限差分',[ugrc[0],ugrc[1]],[.00025377155,.00022738],rtol=3e-5,method='对四个原始参数中央差分，含共同开路量')
matrix=np.array([[0,1,0],[-k/m,-b0/m,G/m],[0,-G/L,-R/L]])
roots,vec=np.linalg.eig(matrix)
ordered=sorted(roots,key=lambda z:z.real)
check('T17','三个特征根实部',[z.real for z in ordered],[-199.498744976,-.3506275118,-.3506275118],method='机电状态矩阵特征值')
check('T17','振动根虚部大小',max(abs(roots.imag)),6.32278128995)
coeff=[m*L,m*R+b0*L,b0*R+k*L+G*G,k*R]
check('T17','特征多项式各根残差',[abs(np.polyval(coeff,z)) for z in roots],[0,0,0],atol=2e-12,method='根代入原始三次多项式')
x0=np.array([.01,0.,0]);W0=k*x0[0]**2/2
P=R*R+b0*R*L/m+k*L*L/m
qe=W0*(G*G*(R+b0*L/m))/(b0*P+G*G*(R+b0*L/m))
check('T17','有限自感焦耳热微焦',qe*1e6,57.12655158)
# 从连续Lyapunov方程解积分，不调用推导所得比例。
def integral_quadratic(Q):
    # M^T H + H M = -Q；行主序展平。
    lhs=np.kron(matrix.T,np.eye(3))+np.kron(np.eye(3),matrix.T)
    HH=np.linalg.solve(lhs,-Q.reshape(-1)).reshape((3,3))
    return float(x0@HH@x0)
Qe=np.diag([0,0,R]);Qm=np.diag([0,b0,0])
heat_e=integral_quadratic(Qe);heat_m=integral_quadratic(Qm)
check('T17','独立状态方程热量积分',heat_e,qe,rtol=1e-10,method='连续Lyapunov矩阵方程，9未知数线性求解')
check('T17','机械热与能量收支',[heat_m,heat_e+heat_m],[W0-qe,W0],rtol=1e-10,method='连续Lyapunov矩阵方程')
ts=np.linspace(0,30,120001);modal=np.linalg.solve(vec,x0)
states=np.real(np.exp(ts[:,None]*roots[None,:])@(vec*modal).T)
ie=states[:,2];vel=states[:,1]
check('T17','时域积分焦耳热',float(trap(R*ie**2,ts)),qe,rtol=2e-7,method='120001点精确模态解积分；与解析热量比例比较')
for n in [17,1800,22000]:
    state=states[n];rate=matrix@state
    dW=m*state[1]*rate[1]+k*state[0]*rate[0]+L*state[2]*rate[2]
    check('T17','瞬时机电能量_'+str(n),dW,-b0*state[1]**2-R*state[2]**2,atol=1e-15,method='状态方程与二次能量导数')
values['T17_Rc_uncertainty']=float(ugrc[1]);values['T17_G_uncertainty']=float(ugrc[0])

bg=.020;Pmax=1.820;Pmin=.220
ratio=math.sqrt((Pmin-bg)/(Pmax-bg));rho=(1-ratio)/(1+ratio);A=(Pmax-bg)/(1+rho)**2
check('T18','幅值系数与入射标度',[rho,A],[.5,.8])
check('T18','反射功率可见度',[rho*rho,2*rho/(1+rho*rho)],[.25,.8])
check('T18','45度极大功率',bg+A*.5*(1+rho)**2,.920)
fitx,X=fit([.1,.15,.2,.25,.3]);cov=np.linalg.inv(X.T@X)*.0005**2
check('T18','最小二乘波长频率',[2*fitx[1],c/(2*fitx[1])],[.1,2.99792458e9],method='五点矩阵最小二乘')
check('T18','斜率协方差频率误差MHz',c/.1*math.sqrt(cov[1,1])/.05/1e6,9.480269926,method='设计矩阵协方差')
check('T18','含共同零点截距误差mm',math.sqrt(cov[0,0]+.002**2)*1000,2.037154879)
check('T18','截距斜率协方差',cov[0,1],-2*.0005**2/10,method='设计矩阵逆')
for z in [.002,.05]:
    fitshift,_=fit(np.array([.1,.15,.2,.25,.3])+z)
    check('T18','共同偏移不改斜率_'+str(z),fitshift[1],.05,method='共同偏移后重新最小二乘')
for xx in [.125,.173]:
    k=2*math.pi/.1
    check('T18','移动坐标零点的相位补偿_'+str(xx),
          math.cos(2*k*xx),math.cos(2*k*(xx+.002)-2*k*.002),method='相位参考变换数值反代')

diagramfile=Path(__file__).resolve().parents[1]/'src/tikz/第25讲_TikZ配图.tex'
line=re.search(r'csname DiagramW12.*?\\else.*?plot coordinates \{([^}]+)\}',diagramfile.read_text(),re.S)[1]
pts=re.findall(r'\((-?\d+\.\d+),(-?\d+\.\d+)\)',line)
for t,ref in [(0,1.220),(2,.626),(4,.032)]:
    actual=next(float(y) for x,y in pts if abs(float(x)-t)<1e-8)
    check('W12','TikZ曲线采样_'+str(t),actual,ref,rtol=1e-6,method='读取实际TikZ坐标，核对0/45/90度实测数据')
groups={}
for rec in records:groups[rec['group']]=groups.get(rec['group'],0)+1
assert set(groups)==set(['W'+str(i) for i in range(1,13)]+['T'+str(i) for i in range(1,19)])
result=dict(count=len(records),count_description='每条记录为一个给定参数实例的一项数值比较；同一题的多个观测、残差或参数实例分别计数，非独立公式数量。积分网格点不单独计数。',
            all_passed=all(rec['passed'] for rec in records),groups=groups,values=values,records=records)
dest=Path(__file__).resolve().parent/'检查'/'第二十五讲_公式核验.json'
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(dict(count=len(records),all_passed=result['all_passed'],values=values,
                     failed=[r for r in records if not r['passed']]),ensure_ascii=False,indent=2))
