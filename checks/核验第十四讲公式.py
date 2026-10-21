from pathlib import Path
import json, math
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
rows=[]
gx,gw=np.polynomial.legendre.leggauss(256)
def quad(fn,a,b):
    x=(b-a)*gx/2+(a+b)/2
    return float((b-a)/2*np.dot(gw,fn(x)))
def check(name,actual,expected,tol=2e-8,kind='积分或恒等式'):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    scale=float(np.max(np.abs(b)))
    error=float(np.max(np.abs(a-b)))/(scale if scale>1e-15 else 1.)
    rows.append(dict(name=name,normalized_error=error,tolerance=tol,kind=kind))
    assert np.isfinite(error) and error<=tol,(name,error,tol,a,b)
def rk(fn,y0,T,n=8000):
    y=np.asarray(y0,dtype=float);h=T/n
    for j in range(n):
        t=j*h;k1=fn(t,y);k2=fn(t+h/2,y+h*k1/2)
        k3=fn(t+h/2,y+h*k2/2);k4=fn(t+h,y+h*k3)
        y+=h*(k1+2*k2+2*k3+k4)/6
    return y

# 从真实面电荷库仑积分求极化球电势与电场，未预设球内场均匀。
phi=(np.arange(256)+.5)*2*math.pi/256
u=gx[:,None];s=np.sqrt(1-u*u)
normals=np.stack(np.broadcast_arrays(s*np.cos(phi),s*np.sin(phi),u),axis=-1).reshape(-1,3)
weights=np.broadcast_to(gw[:,None]*2*math.pi/256,(256,256)).ravel()
def surface_field(point,radius,P,center=np.zeros(3)):
    delta=np.asarray(point)-center-radius*normals
    dist=np.linalg.norm(delta,axis=1)
    dq=(normals@P)*radius**2*weights
    potential=np.sum(dq/dist)/(4*math.pi)
    field=np.sum(dq[:,None]*delta/dist[:,None]**3,axis=0)/(4*math.pi)
    return potential,field
P=np.array([.3,-.4,.8]);a=.9
for r in [[0,0,0],[.1,.3,.2],[-.4,.2,.1],[0,0,.63]]:
    v,E=surface_field(r,a,P)
    check('极化球内部面电荷电势',v,np.dot(P,r)/3,kind='库仑面积分')
    check('极化球内部面电荷电场',E,-P/3,kind='库仑面积分')
for r in [[1.8,0,0],[0,0,2],[.8,1.5,.7],[-1.1,.8,1.6]]:
    r=np.asarray(r);rr=np.linalg.norm(r);p=4*math.pi*a**3*P/3
    v,E=surface_field(r,a,P)
    check('极化球外部偶极电势',v,np.dot(p,r)/(4*math.pi*rr**3),kind='库仑面积分')
    check('极化球外部偶极电场',E,(3*np.dot(p,r)*r/rr**2-p)/(4*math.pi*rr**3),kind='库仑面积分')
check('球面净电荷',np.dot(weights,normals@P),0)
check('球面电荷偶极矩',np.sum((normals@P)[:,None]*a**3*normals*weights[:,None],axis=0),4*math.pi*a**3*P/3)
for a,P0 in [(.7,.9),(1.4,.6),(2.,1.2)]:
    inside=quad(lambda r:4*math.pi*r*r*P0*P0/18,0,a)
    angular=quad(lambda u:1+3*u*u,-1,1)*2*math.pi
    outside=P0**2*a**6/18*angular*quad(lambda t:np.full_like(t,1/a**3)*t*t,0,1)
    total=2*math.pi*a**3*P0**2/9
    check('球内外场能总和',inside+outside,total)
    check('球内场能比例',inside/total,1/3)

# 圆棒两端圆盘的环微元积分，包含非中心轴线点。
for radius,length in [(.6,2.),(1.,3.),(.8,7.)]:
    for z in [0,.2*length,-.3*length]:
        sm=length/2+z;sp=length/2-z
        numeric=-.5*(quad(lambda r:sm*r/(sm*sm+r*r)**1.5,0,radius)+quad(lambda r:sp*r/(sp*sp+r*r)**1.5,0,radius))
        exact=-.5*(2-sm/math.sqrt(sm*sm+radius**2)-sp/math.sqrt(sp*sp+radius**2))
        check('有限圆棒轴线退极化场',numeric,exact,kind='库仑环积分')
for ratio in [30,60,100]:
    N=1-1/math.sqrt(1+4/ratio**2)
    check('长棒中心退极化首项',N,2/ratio**2,tol=4/ratio**2,kind='渐近比较')

# 给定径向极化的体面电荷抵消、Gauss场与全空间场能。
for power in [1,1.5,2,3,5,8]:
    for radius,C in [(.8,.7),(1.3,.4)]:
        volume=quad(lambda r:-4*math.pi*C*(power+2)*r**(power+1),0,radius)
        surface=4*math.pi*C*radius**(power+2)
        check('径向幂律极化总电荷',volume+surface,0)
        r=.61*radius
        Q=quad(lambda q:-4*math.pi*C*(power+2)*q**(power+1),0,r)
        check('径向幂律Gauss内场',Q/(4*math.pi*r*r),-C*r**power)
        energy=quad(lambda r:2*math.pi*C*C*r**(2*power+2),0,radius)
        check('径向幂律极化场能',energy,2*math.pi*C*C*radius**(2*power+3)/(2*power+3))
for a,b,k in [(.7,1.4,2.),(1.,3.,4.),(1.2,2.,1.5)]:
    kap=lambda r:k*(1+r/a)
    volume=quad(lambda r:-(k/a)/kap(r)**2,a,b)
    inner=-(1-1/kap(a));outer=1-1/kap(b)
    check('变介电常数球壳体电荷',volume,1/kap(b)-1/kap(a))
    check('变介电常数球壳净束缚电荷',volume+inner+outer,0)
for k,rho,R in [(2.,.8,.7),(4.,.6,1.2),(8.,.2,2.)]:
    volume=quad(lambda r:-4*math.pi*r*r*(1-1/k)*rho,0,R)
    surface=4*math.pi*R**3*(1-1/k)*rho/3
    check('带自由体电荷球束缚电荷抵消',volume+surface,0)
    check('均匀介质内非零束缚体电荷',volume,-(1-1/k)*4*math.pi*R**3*rho/3)

# 直接解电势和D法向连续矩阵，对照球、圆柱与真实空腔的系数。
for ki,km in [(1,2),(1,5),(1,20),(2,1),(5,1),(8,3),(3,8),(4,4)]:
    for dim in [2,3]:
        A,B=np.linalg.solve([[1,1],[ki,-(dim-1)*km]],[1,km])
        check('介质球柱内场边界方程',A,dim*km/(ki+(dim-1)*km),kind='边界线性方程')
        check('介质球柱外部扰动边界方程',B,(ki-km)/(ki+(dim-1)*km),kind='边界线性方程')
for k in [1,2,5,20]:
    real=3*k/(1+2*k);lorentz=(k+2)/3
    check('真实空腔与冻结空腔差',lorentz-real,2*(k-1)**2/(3*(1+2*k)))

# 椭球无穷积分，变量替换s=(t/(1-t))²使两端平滑。
def depol(axes):
    axes=np.asarray(axes);product=float(np.prod(axes))
    def integrand(t,i):
        s=(t/(1-t))**2;ds=2*t/(1-t)**3
        return product/2*ds/((s+axes[i]**2)*np.sqrt(np.prod(s[:,None]+axes**2,axis=1)))
    return np.array([quad(lambda t:integrand(t,i),0,1) for i in range(3)])
for axes in [[1,1,1],[.7,1.3,2.4],[1,1,4],[3,3,.6],[.6,2,1.2]]:
    N=depol(axes)
    check('椭球退极化因子迹',N.sum(),1,kind='椭球积分')
    check('椭球退极化尺度不变',depol(np.array(axes)*1.7),N,kind='椭球积分')
    check('椭球退极化轴置换',depol(np.array(axes)[[2,0,1]]),N[[2,0,1]],kind='椭球积分')
check('球形椭球因子',depol([1,1,1]),np.full(3,1/3),kind='椭球积分')
for long in [1.4,2,4,8]:
    e=math.sqrt(1-1/long**2)
    Nz=(1-e*e)/(2*e**3)*(math.log((1+e)/(1-e))-2*e)
    check('长旋转椭球因子闭式',depol([1,1,long]),[(1-Nz)/2,(1-Nz)/2,Nz],kind='椭球积分')
for N in [0,.1,1/3,.5,.8,1]:
    for chi in [.2,1,4]:
        P=chi/(1+N*chi)
        check('极化自洽内场',P,chi*(1-N*P))
        internal=P*P/(2*chi);depolar=N*P*P/2
        check('自洽极化极小能量',internal+depolar-P,-P/2)

# 有限插入电容，固定电荷/电压的能量导数和电源功。
for k,x in [(2,.2),(4,.5),(8,.7)]:
    C=lambda x:1+(k-1)*x;h=1e-5;Q=.8;U=1.3
    derivative=(C(x+h)-C(x-h))/(2*h)
    Fq=-(Q*Q/(2*C(x+h))-Q*Q/(2*C(x-h)))/(2*h)
    Fu=(.5*U*U*C(x+h)-.5*U*U*C(x-h))/(2*h)
    check('孤立电容器插入力',Fq,Q*Q*(k-1)/(2*C(x)**2),kind='有限差分')
    check('恒压电容器插入力',Fu,U*U*(k-1)/2,kind='有限差分')
    source=U*U*(C(x)-C(0));field=.5*U*U*(C(x)-C(0))
    mechanical=quad(lambda q:np.full_like(q,.5*U*U*(k-1)),0,x)
    check('恒压插入供能',source,field+mechanical)
for k1,k2,d1,d2,U in [(2,4,.4,.6,1),(6,2,.3,.8,2),(1.5,8,.9,.4,.6)]:
    D=U/(d1/k1+d2/k2);P1=(1-1/k1)*D;P2=(1-1/k2)*D
    E1=D/k1;E2=D/k2
    check('双层电容器总电压',d1*E1+d2*E2,U)
    check('双层界面总面电荷',E2-E1,P1-P2)
    check('双层场能与端口能量',(d1*D*E1+d2*D*E2)/2,D*U/2)
for t,d,P0 in [(.4,1,.8),(.3,.9,1.2),(.7,1.4,.6)]:
    D=P0*t/d;Eair=D;Eslab=D-P0
    check('永久极化片短接电压',Eair*(d-t)+Eslab*t,0)
    check('永久极化片短接板电荷',D,P0*t/d)
eps=np.array([[4,1],[1,2.]])
E=np.array([1.,-eps[1,0]/eps[1,1]]);D=eps@E;P=D-E
check('各向异性板D法向',D[1],0,kind='张量方程')
check('各向异性板E和P',np.r_[E,P],[1,-.5,2.5,.5],kind='张量方程')
check('各向异性介电本征值',np.linalg.eigvalsh(eps),[3-math.sqrt(2),3+math.sqrt(2)],kind='张量方程')
check('各向异性极化能正性',P@E,2.25,kind='张量方程')

# 取向分布的角积分，避免小x时相减损失。
for x in [.01,.1,.5,1,3,8,20]:
    Z=quad(lambda u:np.exp(x*u),-1,1)
    avg=quad(lambda u:u*np.exp(x*u),-1,1)/Z
    check('朗之万取向角积分',avg,1/math.tanh(x)-1/x,kind='统计角积分')
    check('取向配分积分',Z,2*math.sinh(x)/x,kind='统计角积分')
for x in [.001,.003,.01]:
    avg=quad(lambda u:u*np.exp(x*u),-1,1)/quad(lambda u:np.exp(x*u),-1,1)
    check('取向弱场立方修正',avg,x/3-x**3/45,tol=1e-8,kind='渐近比较')
for x in [.05,.2,.5,.8]:
    kap=(1+2*x)/(1-x);chi=kap-1;local=1+chi/3
    check('克劳修斯莫索蒂局域场自洽',3*x*local,chi)
    check('克劳修斯莫索蒂反馈参数',(kap-1)/(kap+2),x)
z=np.array([0.,0.,1]);xx=np.array([1.,0.,0.])
for P1,P2,n,expected in [(z,-z,xx,0),(z,-z,z,2),(xx,z,(xx+z)/math.sqrt(2),0),(xx,z,(xx-z)/math.sqrt(2),math.sqrt(2))]:
    check('畴壁面电荷投影',np.dot(P1-P2,n),expected)
for phase in [.1,.3,.8,1.2]:
    Em=.8;Pm=.6;om=1.4;period=2*math.pi/om
    work=quad(lambda t:Em*np.cos(om*t)*(-om*Pm*np.sin(om*t-phase)),0,period)
    check('极化滞后周期损耗',work,math.pi*Em*Pm*math.sin(phase),kind='周期积分')
    check('极化滞后平均功率',work/period,.5*om*Em*Pm*math.sin(phase),kind='周期积分')
    total=quad(lambda t:Em*np.cos(om*t)*(-om*Em*np.sin(om*t)-om*Pm*np.sin(om*t-phase)),0,period)
    check('电位移电流周期功',total,work,kind='周期积分')

# 偏心空腔通过两套真正的面电荷积分验证。
P=np.array([.3,-.4,.8]);c=np.array([.35,.1,.05]);a=.25;R=1.3
for offset in [[0,0,0],[.04,.02,.07],[-.07,.06,0]]:
    point=c+offset
    _,large=surface_field(point,R,P);_,removed=surface_field(point,a,-P,c)
    check('偏心冻结极化空腔取消',large+removed,0,kind='库仑面积分')
    bigmoment=np.sum((normals@P)[:,None]*R**3*normals*weights[:,None],axis=0)
    smallmoment=np.sum((normals@(-P))[:,None]*a*a*(c+a*normals)*weights[:,None],axis=0)
    check('偏心空腔总偶极矩',bigmoment+smallmoment,4*math.pi*(R**3-a**3)*P/3)

# 包覆球：对照四未知数边界矩阵，包括均匀、零核心、零壳厚极限。
for k1,k2 in [(4,2),(2,6),(1,4),(7,1),(3,3)]:
    for ratio in [.3,.5,.8,1.]:
        a=ratio;b=1.;f=a**3
        mat=[[1,-1,1/a**3,0],[k1,-k2,-2*k2/a**3,0],[0,1,-1/b**3,1/b**3],[0,k2,2*k2/b**3,-2/b**3]]
        sol=np.linalg.solve(mat,[0,0,1,1]);h=(k1-k2)/(k1+2*k2)
        B=3/(k2+2+2*f*h*(k2-1));C=a**3*h*B;A=(1-h)*B;G=1-(1-f*h)*B
        check('包覆球四边界系数',sol,[A,B,C,G],kind='边界线性方程')
        delta=(k2+2)*(k1+2*k2)+2*f*(k2-1)*(k1-k2)
        alphared=((k2-1)*(k1+2*k2)+f*(k1-k2)*(1+2*k2))/delta
        check('包覆球总极化率',G,alphared)
        check('包覆球内界面净极化面电荷',(k1-1)*A-(k2-1)*(B+2*C/a**3),B+2*C/a**3-A)
        if ratio==1:check('包覆球零壳厚极限',[A,G],[3/(k1+2),(k1-1)/(k1+2)])
h=.25;f=.125;B=3/(4+2*f*h);A=(1-h)*B;G=1-(1-f*h)*B
check('综合题包覆球数值',[A,B,G],[36/65,48/65,37/130])

# 诱导椭球平动与转动：能量导数、p乘场梯度和运动方程分别比较。
apar=1.2;aper=.6;mass=.9;J=.8;cQ=.7
for z in [1.5,2,3]:
    for theta in [.2,.6,1.2]:
        AA=apar*math.cos(theta)**2+aper*math.sin(theta)**2
        E=cQ/z**2;U=lambda z,theta:-.5*(apar*math.cos(theta)**2+aper*math.sin(theta)**2)*(cQ/z**2)**2
        h=1e-5
        F=-(U(z+h,theta)-U(z-h,theta))/(2*h)
        torque=-(U(z,theta+h)-U(z,theta-h))/(2*h)
        check('诱导椭球平动力能量导数',F,-2*AA*E*E/z,kind='有限差分')
        check('诱导椭球平动力偶极梯度',F,(AA*E)*(-2*E/z),kind='有限差分')
        check('诱导椭球转矩能量导数',torque,-(apar-aper)*E*E*math.sin(theta)*math.cos(theta),kind='有限差分')
for zi,T in [(3.,1.5),(2.5,1.),(2.,.6)]:
    def motion(t,y):
        z,v=y;return np.array([v,-2*apar*cQ*cQ/(mass*z**5)])
    z,v=rk(motion,[zi,0],T)
    speed=cQ*math.sqrt(apar/mass*(z**-4-zi**-4))
    check('椭球落向点电荷速度',-v,speed,kind='运动方程积分')
    check('椭球平动机械能',mass*v*v/2-apar*cQ*cQ/(2*z**4),-apar*cQ*cQ/(2*zi**4),kind='运动方程积分')
for z in [1.5,2,3]:
    om=cQ/z**2*math.sqrt((apar-aper)/J);T=.8/om;angle0=.002
    def rotational(t,y):
        theta,v=y;return np.array([v,-om*om*math.sin(theta)*math.cos(theta)])
    angle,v=rk(rotational,[angle0,0],T)
    check('椭球小角振动近似',[angle,v],[angle0*math.cos(om*T),-angle0*om*math.sin(om*T)],tol=5*angle0**2,kind='小角近似')

# 书面SI数值按有效位数比较；与单位化精确关系分开统计。
eps0=8.854e-12;kb=1.380649e-23
S=.020;d=.001;k=4;U0=200;C0=eps0*S/d;C=k*C0;Q0=C0*U0
check('电容器SI数值',[C0*1e12,C*1e12,Q0*1e9,C*U0*1e9],[177.08,708.32,35.416,141.664],tol=1e-6,kind='数值代入与有效位数')
check('孤立介质插入SI数值',[U0/k,(U0/k)/d,eps0*(k-1)*U0/(k*d)*1e6],[50,50000,1.3281],tol=1e-6,kind='数值代入与有效位数')
check('供能源和场能SI数值',[(C0*U0**2/2)*1e6,(Q0**2/(2*C))*1e6,(C*U0**2/2)*1e6,U0*(C*U0-Q0)*1e6],[3.5416,.8854,14.1664,21.2496],tol=1e-6,kind='数值代入与有效位数')
B=(.005-.003)/(1/300-1/600);A=.003-B/600;n=1e25
alpha=eps0*A/n;p0=math.sqrt(3*eps0*kb*B/n)
check('温度极化数据参数',[A,B],[.001,1.2],kind='数值代入与有效位数')
check('分子位移极化率数值',alpha*1e40,8.85,tol=.001,kind='数值代入与有效位数')
check('永久分子偶极矩数值',p0*1e30,6.63,tol=.001,kind='数值代入与有效位数')

groups={k:dict(count=sum(r['kind']==k for r in rows),maximum_normalized_error=max(r['normalized_error'] for r in rows if r['kind']==k)) for k in Counter(r['kind'] for r in rows)}
approx={'渐近比较','小角近似','数值代入与有效位数'}
result=dict(count=len(rows),all_passed=True,maximum_normalized_error=max(r['normalized_error'] for r in rows),exact_maximum_normalized_error=max(r['normalized_error'] for r in rows if r['kind'] not in approx),kinds=dict(Counter(r['kind'] for r in rows)),groups=groups,checks=rows,count_description='同一关系在不同参数下的实例，包含积分、边界线性方程、有限差分、运动方程及近似和有效位数比较；不代表独立公式数量。')
(ROOT/'检查'/'第十四讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2))
