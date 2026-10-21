"""每条记录为一个参数实例的一次数值比较；不将积分网格点计入次数。"""
from pathlib import Path
import numpy as np
import math,json
mu=4*math.pi*1e-7;e=1.602176634e-19;m=9.1093837139e-31
h=6.62607015e-34;hb=h/(2*math.pi);kb=1.380649e-23;c=299792458.
eps=1/(mu*c*c);phi=h/(2*e);rows=[]
trap=np.trapz
def check(g,n,a,b,rtol=2e-5,atol=1e-14,method='独立数值代入'):
    x=np.asarray(a);y=np.asarray(b);err=float(np.max(abs(x-y)))
    tol=atol+rtol*float(np.max(abs(y)))
    rows.append(dict(group=g,name=n,computed=x.tolist(),reference=y.tolist(),
        absolute_error=err,tolerance=tol,method=method,passed=err<=tol))
def d(f,x,hh):return (f(x+hh)-f(x-hh))/(2*hh)
def dd(f,x,hh):return (f(x+hh)+f(x-hh)-2*f(x))/(hh*hh)
def integrate(f,lo,hi,n=100001):
    x=np.linspace(lo,hi,n);return float(trap(f(x),x))
A=e*e/m
# W1: 界面守恒与两侧总电流；三个真实时刻分别比较。
ta=6*eps/3e-8
E1=lambda t:1e4/3*(1+np.exp(-t/ta));E2=lambda t:1e4/3*(2-np.exp(-t/ta))
sf=lambda t:4*eps*E2(t)-2*eps*E1(t)
check('W1','时间常数ms',ta*1e3,1.77084,rtol=1e-4)
for t in [0,ta,3*ta]:
    check('W1','面连续性_'+str(t),d(sf,t,1e-8),2e-8*E1(t)-1e-8*E2(t),method='中央差分')
    check('W1','总电流连续_'+str(t),2e-8*E1(t)+2*eps*d(E1,t,1e-8),1e-8*E2(t)+4*eps*d(E2,t,1e-8),method='中央差分')
check('W1','末态电荷',sf(50*ta),1.77084e-7,rtol=1e-4)
# W2: 空间与时间积分，三个相位检查相同能量。
L=.15;E0=100;k=math.pi/L;omega=c*k
check('W2','基频Hz',omega/(2*math.pi),9.993081933e8)
for ph in [0,.6,math.pi/2]:
    energy=integrate(lambda x:.5*eps*E0**2*(np.sin(k*x)**2*math.cos(ph)**2+np.cos(k*x)**2*math.sin(ph)**2),0,L)
    check('W2','空间能量积分_'+str(ph),energy,eps*E0**2*L/4,method='100001点空间积分')
check('W3','开路电势差',-2*.5*.1,-.1)
check('W3','机械焦耳功率',.5*.1*.5*2,.5**2*.2)
# W4: 两种密度约定和温度代入。
ns=5e28*15/16
la=math.sqrt(1/(mu*ns*A))
check('W4','电子与对约定',la**2,(2*m)/(mu*(ns/2)*(2*e)**2))
check('W4','4K穿透nm',la*1e9,24.5,rtol=.002)
check('W4','近临界放大',1/math.sqrt(1-.99**4),5.03767,rtol=2e-4)
# W5: 一个周期时间积分独立检验正常损耗和超流净功。
sn=1e27*A*1e-13/(1+1j*.01);ss=1e28*A/(1j*1e11)
th=np.linspace(0,2*math.pi,100001)
for name,sg in [('正常',sn),('超流',ss)]:
    p=float(trap(np.real(sg*.01*np.exp(1j*th))*.01*np.cos(th),th)/(2*math.pi))
    check('W5',name+'功率时间积分',p,.5*sg.real*.01**2,atol=2e-10,method='100001点周期积分')
check('W5','正常电流相位度',-np.angle(sn)*180/math.pi,.5729387)
# W6: 电流与两项能量分别积分。
la=50e-9;B=.01
check('W6','片电流积分',integrate(lambda x:B/(mu*la)*np.exp(-x/la),0,30*la),B/mu,method='100001点积分')
for name,f in [('磁能',lambda x:B**2*np.exp(-2*x/la)/(2*mu)),('动能',lambda x:mu*la**2*(B*np.exp(-x/la)/(mu*la))**2/2)]:
    check('W6',name,integrate(f,0,30*la),B*B*la/(4*mu),method='100001点积分')
# W7: 有限差分伦敦方程、边界与极值。
bfun=lambda x:2*np.cosh(x)/np.cosh(1)+np.sinh(x)/np.sinh(1)
check('W7','两面边界',[bfun(-1),bfun(1)],[1,3])
for x in [-.5,0,.5]:
    check('W7','二阶方程_'+str(x),dd(bfun,x,1e-3),bfun(x),rtol=1e-6,method='中央二阶差分')
xx=math.atanh(-.5/math.tanh(1))
check('W7','换向位置',xx,-.787,rtol=.003)
check('W7','换向点导数',d(bfun,xx,1e-5),0,atol=1e-9,method='中央差分')
# W8: 电感与频率。
lk=mu*1e-14*.001/(1e-6*20e-9)
check('W8','动能电感nH',lk*1e9,.62831853)
check('W8','频率GHz',1/(2*math.pi*math.sqrt((lk+.4e-9)*1e-12))/1e9,4.963,rtol=.001)
# W9: 三个实际历史实例。
for f,n,target in [(.35,0,.07),(.6,1,.92),(.6,0,.12)]:
    I=(n-f)*phi/1e-10;ph=f*phi+80e-12*I
    check('W9','孔磁通_'+str((f,n)),ph/phi,target)
    check('W9','磁通子_'+str((f,n)),(ph+20e-12*I)/phi,n,atol=1e-14)
# W10: 热力学导数独立差分。
free=lambda T:-50*(T-8)**2
check('W10','熵差',-d(free,7,1e-4),-100,method='温度中央差分')
check('W10','热容差',-7*dd(free,7,1e-3),700,rtol=1e-7,method='温度二阶差分')
check('W10','临界场T',math.sqrt(mu*100),.0112,rtol=.002)
# W11: 对束缚与BCS自洽方程分别积分。
u=.3;O=20;bind=2*O/math.expm1(2/u);gap=O/math.sinh(1/u)
check('W11','库珀积分',u*integrate(lambda x:1/(2*x+bind),0,O,300001),1,rtol=2e-7,method='300001点积分')
check('W11','BCS零温积分',u*integrate(lambda x:1/np.sqrt(x*x+gap*gap),0,O),1,rtol=1e-8,method='100001点积分')
check('W11','束缚能meV',bind,.0510,rtol=.002)
check('W11','能隙meV',gap,1.43,rtol=.002)
# W12: BdG矩阵对角化及占据。
z=np.linspace(-35,0,300001);xx=np.exp(z)
for coupling in [.2,.3,.4]:
    lo,hi=1e-7,.2
    for _ in range(60):
        mid=(lo+hi)/2
        F=coupling*float(trap(np.tanh(xx/(2*mid)),z))
        if F>1:lo=mid
        else:hi=mid
    tc=(lo+hi)/2
    check('W11','有限温度临界方程_'+str(coupling),coupling*float(trap(np.tanh(xx/(2*tc)),z)),1,rtol=1e-9,method='对数变量300001点积分及二分求根')
    approx=2*math.exp(np.euler_gamma)/math.pi*math.exp(-1/coupling)
    check('W11','临界温度渐近式_'+str(coupling),tc,approx,rtol=1e-5,method='数值求根与弱耦合渐近式')
gap=1.764*kb*9.2
check('W12','零温能隙meV',gap/e*1e3,1.4,rtol=.003)
check('W12','破对频率GHz',2*gap/h/1e9,676,rtol=.002)
check('W12','相干尺度nm',hb*1e6/(math.pi*gap)*1e9,150,rtol=.002)
for xi in [0,1,-1]:
    M=np.array([[xi,1],[1,-xi]],float);val,vec=np.linalg.eigh(M)
    check('W12','BdG本征能_'+str(xi),val,[-math.sqrt(xi*xi+1),math.sqrt(xi*xi+1)],method='numpy对称矩阵对角化')
    check('W12','电子占据_'+str(xi),abs(vec[0,0])**2,.5*(1-xi/math.sqrt(xi*xi+1)),method='本征向量权重')

# T1: 场与源的偏导数相容性，归一化真空参数。
for x,t in [(.2,.4),(.8,.6),(1.1,1.2)]:
    rho=lambda xx,tt:np.cos(xx)*np.cos(tt)
    j=lambda xx,tt:np.sin(xx)*np.sin(tt)
    Ef=lambda xx,tt:np.sin(xx)*np.cos(tt)
    check('T1','源连续性_'+str((x,t)),d(lambda z:rho(x,z),t,1e-5)+d(lambda z:j(z,t),x,1e-5),0,atol=1e-9,method='两变量中央差分')
    check('T1','安培右侧_'+str((x,t)),j(x,t)+d(lambda z:Ef(x,z),t,1e-5),0,atol=1e-9,method='中央差分')
# T2: 独立联立界面方程，不从结果反填系数。
Z0=mu*c;Z2=Z0/2
rr,tt=np.linalg.solve(np.array([[1,-1],[-1/Z0,-1/Z2]]),[-1,-1/Z0])
check('T2','线性边界解',[rr,tt],[-1/3,2/3],method='numpy线性方程求解')
check('T2','功率守恒',rr*rr+Z0/Z2*tt*tt,1)
check('T2','三束功率',[900/(2*Z0),900*rr*rr/(2*Z0),900*tt*tt/(2*Z2)],[1.194,.133,1.062],rtol=.004)
# T3: 面电流空间散度及面电荷时间变化。
K0=2e-6;kk=20;tau=.005
sf=lambda y,t:-kk*K0*tau*(1-np.exp(-t/tau))*np.cos(kk*y)
K=lambda y,t:K0*np.sin(kk*y)*np.exp(-t/tau)
for y,t in [(0,.001),(.05,.004),(.1,.01)]:
    check('T3','面守恒_'+str((y,t)),d(lambda z:sf(y,z),t,1e-8)+d(lambda z:K(z,t),y,1e-6),0,atol=1e-12,method='时空中央差分')
check('T3','末态幅值',kk*K0*tau,2e-7)
# T4: 皮深与阻抗，复根反代。
om=2*math.pi*1e6;sig=5.8e7;delta=math.sqrt(2/(mu*sig*om))
check('T4','铜皮深微米',delta*1e6,66.1,rtol=.001)
check('T4','铜表面电阻mΩ',1/(sig*delta)*1e3,.261,rtol=.002)
check('T4','伦敦表面电抗μΩ',om*mu*50e-9*1e6,.395,rtol=.001)
# T5: 脉冲积分和功的独立积分。
As=4e28*A;Ep=.001;Tp=1e-9
jfun=lambda t:As*Ep*Tp/math.pi*(1-np.cos(math.pi*t/Tp))
energy=integrate(lambda t:jfun(t)*Ep*np.sin(math.pi*t/Tp),0,Tp)
check('T5','脉冲后电流',jfun(Tp),7.18e8,rtol=.002)
check('T5','脉冲总功',energy,jfun(Tp)**2/(2*As),rtol=1e-8,method='100001点时间积分')
# T6: 展开误差与逆向温度误差传播。
for t in [.95,.99]:
    eta=1-t;ratio=math.sqrt(1-t**4)/(2*math.sqrt(eta))
    check('T6','近似相对误差_'+str(t),ratio-1,-.75*eta+7/32*eta**2,rtol=.001,method='精确比值与二阶展开')
t=.99;ll=lambda t:1/math.sqrt(1-t**4)
check('T6','对数灵敏度',d(lambda t:math.log(ll(t)),t,1e-7),2*t**3/(1-t**4),method='温度中央差分')
check('T6','温度不确定度',.01/(2*t**3/(1-t**4)),2.03e-4,rtol=.002)
# T7: 三个实际厚度，比对总能量的积分和边界表达。
for aa in [.1,1,3]:
    bf=lambda x:np.sinh(x)/math.sinh(aa)
    jf=lambda x:-np.cosh(x)/(mu*math.sinh(aa))
    check('T7','片电流_'+str(aa),integrate(jf,-aa,aa),-2/mu,rtol=1e-8,method='100001点截面积分')
    U=integrate(lambda x:bf(x)**2/(2*mu)+mu*jf(x)**2/2,-aa,aa)
    check('T7','总储能_'+str(aa),U,1/(mu*math.tanh(aa)),rtol=1e-8,method='100001点双项能量积分')
# T8: 自洽薄膜方程反代。
la=200e-9;thick=40e-9;k=1e5;La=2*la*la/thick;Aext=1e-8
K=-2*k*Aext/(mu*(1+k*La));Atot=Aext+mu*K/(2*k)
check('T8','薄膜伦敦关系',K,-thick*Atot/(mu*la*la))
check('T8','法向场比例',Atot/Aext,1/6)
# T9: 温度变化的电流和磁通。
for t in [0,.8]:
    Lk=10e-12/(1-t**4);Ltot=90e-12+Lk
    I=-.25*phi/Ltot;actual=.25*phi+90e-12*I
    check('T9','量子约束_'+str(t),(actual+Lk*I)/phi,0,atol=1e-14)
    check('T9','电流μA_'+str(t),I*1e6,(-5.17 if t==0 else -4.83),rtol=.002)
check('T9','末态孔磁通',actual/phi,.0396,rtol=.002)
# T10: 对实际微波振幅和三个相位数值求周期平均。
a=2e-6/(phi*1e10);J1=integrate(lambda t:np.cos(t-a*np.sin(t)),0,2*math.pi)/(2*math.pi)
check('T10','贝塞尔三阶近似',J1,a/2-a**3/16,rtol=1e-6,method='100001点周期积分')
for phase in [0,.7,math.pi/2]:
    av=integrate(lambda t:5e-6*np.sin(phase+t+a*np.sin(t)),0,2*math.pi)/(2*math.pi)
    check('T10','直流超流_'+str(phase),av,-5e-6*J1*math.sin(phase),atol=1e-18,method='100001点周期积分')
# T11: 对共模相位扫描，求两个结总电流最大值。
angles=np.linspace(0,2*math.pi,100001)
for f in [0,.25,.5]:
    i=5*np.sin(angles+math.pi*f)+3*np.sin(angles-math.pi*f)
    check('T11','相位扫描最大μA_'+str(f),float(np.max(i)),math.sqrt(34+30*math.cos(2*math.pi*f)),rtol=1e-8,method='100001点相位扫描')
# T12: 极值、密度与临界电流。
jdim=lambda u:u*(1-u*u)
check('T12','破对点导数',d(jdim,1/math.sqrt(3),1e-5),0,atol=2e-10,method='中央差分')
check('T12','破对电流密度',phi/(3*math.sqrt(3)*math.pi*mu*(200e-9)**2*20e-9),1.26e11,rtol=.003)
for u in [.2,.6]:
    check('T12','归一化电流_'+str(u),jdim(u)/(2/(3*math.sqrt(3))),(.499 if u==.2 else .998),rtol=.001)
# T13: 空间二阶差分以及恢复能量积分。
ff=lambda x:np.tanh(x/math.sqrt(2))
for x in [.5,2,4]:
    check('T13','GL空间残差_'+str(x),dd(ff,x,1e-3)+ff(x)-ff(x)**3,0,atol=2e-7,method='中央二阶差分')
check('T13','九成恢复位置',math.sqrt(2)*math.atanh(.9),2.08,rtol=.002)
check('T13','表面自由能积分',integrate(lambda x:1/np.cosh(x/math.sqrt(2))**4,0,20),2*math.sqrt(2)/3,rtol=1e-8,method='100001点空间积分')
# T14: 同位素变化及耦合敏感性。
check('T14','同位素Tc比',math.sqrt(100/104),.98058,rtol=1e-5)
check('T14','耦合变化Tc比',math.exp(4-1/.255),1.0816)
ut=lambda M:.25+.001*np.log(M)
Tc=lambda M:M**(-.5)*np.exp(-1/ut(M))
check('T14','质量导数指数',-d(lambda t:np.log(Tc(np.exp(t))),0,1e-4),.5-.001/.25**2,method='对数质量中央差分')
# T15: 对角化的占据和隙边奇异积分的变量代换。
check('T15','两侧电子占据',[.5*(1-2/math.sqrt(5)),.5*(1+2/math.sqrt(5))],[.05279,.94721],rtol=.0001)
check('T15','谱态数积分',integrate(np.cosh,0,math.acosh(2)),math.sqrt(3),rtol=1e-8,method='E=Δcosh(u)变量代换后积分')
# T16: 逐条涡旋热与体电功率。
B=.2;j=1e8;eta=phi*2/1e-7;v=j*phi/eta
check('T16','涡旋速度',v,5)
check('T16','电阻率',B*v/j,1e-8,atol=0)
check('T16','黏滞热与电功率',B/phi*eta*v*v,j*B*v)
# T17: 精确复阻抗、展开和完整损耗积分。
la=100e-9;sn=1e6;om=2*math.pi*1e9;Ba=.001
q=mu*om*sn*la*la;kap=np.sqrt(1/la**2+1j*mu*om*sn);Z=1j*mu*om/kap
check('T17','小q表面电阻',Z.real,mu*om*la*q/2,rtol=1e-8,atol=0)
check('T17','表面电阻印刷数值',Z.real,3.12e-8,rtol=.002,atol=0)
check('T17','小q表面电抗',Z.imag,mu*om*la*(1-3*q*q/8),rtol=1e-10,atol=0)
power=.5*Z.real*(Ba/mu)**2;E0=1j*om*Ba/kap
heat=integrate(lambda x:.5*sn*abs(E0)**2*np.exp(-2*kap.real*x),0,30/kap.real)
check('T17','体内热积分与表面能流',heat,power,rtol=2e-7,atol=0,method='100001点空间积分')
check('T17','每平方米输入功率',power,.00987,rtol=.003,atol=0)
for x in [0,la,3*la]:
    Ef=lambda x:E0*np.exp(-kap*x);Bf=lambda x:Ba*np.exp(-kap*x)
    lhs=d(Ef,x,la*1e-5);rhs=-1j*om*Bf(x)
    check('T17','法拉第复残差_'+str(x),[lhs.real,lhs.imag],[rhs.real,rhs.imag],method='复函数中央差分')
# T18: 势能一阶、二阶导数及相位滑移的支能量比较。
L=1e-10;G=.1*phi/1e-6;U=lambda x:(.4*phi+G*x)**2/(2*L)
I=-.4*phi/L
check('T18','力的势能导数',-d(U,0,1e-10),I*G,rtol=1e-7,atol=0,method='位移中央差分')
kem=G*G/L
check('T18','刚度二阶差分',dd(U,0,1e-9),kem,rtol=1e-6,atol=0,method='位移二阶差分')
check('T18','小振荡频率',math.sqrt((5e-10+kem)/1e-15)/(2*math.pi),153,rtol=.002)
fstar=L*12e-6/phi;xs=(fstar-.4)*phi/G
check('T18','首次滑移微米',xs*1e6,1.80,rtol=.002)
after=(1-fstar)*phi/L
check('T18','滑移后电流μA',after*1e6,8.68,rtol=.001)
change=.5*L*(after*after-(12e-6)**2)
check('T18','滑移支能量差',change,phi*phi/(2*L)*(1-2*fstar),atol=0)
check('T18','释放能量J',-change,3.43e-21,rtol=.003,atol=0)

assert set(x['group'] for x in rows)=={f'W{i}' for i in range(1,13)}|{f'T{i}' for i in range(1,19)}
out=dict(count=len(rows),count_description='一项=一次参数实例的一次数值比较；30题均有记录，积分网格点不计数；次数不等于独立公式数量。',
    all_passed=all(x['passed'] for x in rows),records=rows)
p=Path(__file__).parent/'检查'/'第二十六讲_公式核验.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(dict(count=out['count'],all_passed=out['all_passed'],failures=[x for x in rows if not x['passed']]),ensure_ascii=False))
assert out['all_passed']
