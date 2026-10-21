from pathlib import Path
import numpy as np, math,json
R=Path(__file__).resolve().parent
h=6.62607015e-34;e=1.602176634e-19;hb=h/(2*np.pi);m=9.1093837139e-31;c=299792458;Phi=h/e
records=[]
def ck(group,name,actual,reference,rtol=3e-4,atol=1e-8,method='独立参数代入'):
    a=np.asarray(actual);b=np.asarray(reference)
    err=float(np.max(np.abs(a-b)));tol=float(atol+rtol*np.max(np.abs(b)))
    def serial(z):
        z=np.asarray(z)
        if np.iscomplexobj(z):return {'real':z.real.tolist(),'imag':z.imag.tolist()}
        return z.tolist()
    records.append(dict(group=group,name=name,computed=serial(a),reference=serial(b),
        absolute_error=err,tolerance=tol,method=method,passed=err<=tol))
def integ(f,a,b,n=20001):
    x=np.linspace(a,b,n);return np.trapz(f(x),x)
def v(K):return math.sqrt(2*K*e/m)
def lam(K):return h/(m*v(K))
def ER(R):return hb**2/(2*m*R**2)
def gaussian_phase(sig):
    return integ(lambda z:np.exp(-z*z/2)*np.exp(1j*sig*z)/math.sqrt(2*math.pi),-9,9)
def fd(f,x,step=1e-5):return (f(x+step)-f(x-step))/(2*step)
# Independent gauge covariance in dimensionless units, hbar=m=q=1.
for x in [-.8,-.2,.3,.9]:
    psi=lambda z:(1+.2*z*z)*np.exp(1j*(.7*z+.2*z*z))
    chi=lambda z:.3*z**3
    A=lambda z:.4+.1*z
    pp=lambda z:np.exp(1j*chi(z))*psi(z)
    orig=-1j*fd(psi,x)-A(x)*psi(x)
    after=-1j*fd(pp,x)-(A(x)+.9*x*x)*pp(x)
    ck('理论','协变动量 x='+str(x),after,np.exp(1j*chi(x))*orig,rtol=1e-7,method='有限差分求导')
    ck('理论','概率流 x='+str(x),np.real(np.conj(pp(x))*after),np.real(np.conj(psi(x))*orig),rtol=1e-7,method='有限差分求导')
ck('W1','负电荷相位rad',-e*(Phi/4)/hb,-np.pi/2)
ck('W1','相量',np.exp(-1j*e*Phi/(4*hb)),-1j)
ck('W1','周期Wb/1e-15',Phi/1e-15,4.135667696)
a=100e-9;B=.1;flux=np.pi*a*a*B
ck('W2','磁通Wb/1e-15',flux/1e-15,3.1416)
for r,val in [(200e-9,2.5),(400e-9,1.25)]:
    A=flux/(2*np.pi*r)
    ck('W2','矢势Tm/1e-9 r='+str(r),A/1e-9,val)
    ck('W2','圆周独立积分 r='+str(r),integ(lambda t:np.ones_like(t)*A*r,0,2*np.pi)/flux,1,method='圆周积分')
ck('W2','电子相位rad',-e*flux/hb,-4.773)
ck('W3','波长nm',lam(100)*1e9,.12264)
df=lam(100)/1e-6
ck('W3','条纹mm',df*1e3,.12264);ck('W3','位移mm',.3*df*1e3,.03679)
for y in [-3*df,-df,.3*df,df,3*df]:
    # Avoid subtracting nearly equal square roots.
    r1=np.sqrt(1+(y+5e-7)**2);r2=np.sqrt(1+(y-5e-7)**2)
    exact=2*y*1e-6/(r1+r2)
    ck('W3','近轴路长差 y/df='+str(y/df),exact/(1e-6*df),y/df,rtol=1e-6,method='精确路长')
for A0,kappa,l in [(1.2,.3,1),(-.7,-.5,2)]:
    I1=integ(lambda x:A0*np.ones_like(x),0,l)+integ(lambda y:kappa*l*np.ones_like(y),0,l)
    I2=integ(lambda y:np.zeros_like(y),0,l)+integ(lambda x:(A0+kappa*l)*np.ones_like(x),0,l)
    ck('W4','开放积分1 '+str(l),I1,A0*l+kappa*l*l,method='逐段积分')
    ck('W4','开放积分差 '+str(l),I1-I2,0,method='逐段积分')
for f,ref in [(0,9),(.25,5),(.5,1),(1,9)]:
    ck('W5','相量模平方 f='+str(f),abs(2*np.exp(-2j*np.pi*f)+1)**2,ref,method='直接复数求模')
ck('W5','可见度',(9-1)/(9+1),.8)
ck('W6','高斯相位积分',gaussian_phase(2*np.pi*.1),.8209,method='高斯数值积分')
sn=math.sqrt(-2*math.log(.9))/(2*np.pi)
ck('W6','噪声比上限',sn,.07306);ck('W6','Wb/1e-16',sn*Phi/1e-16,3.0216)
ck('W7','标势脉冲相位',e/hb*integ(lambda t:np.ones_like(t)*40e-9,0,20e-9),1.2154,method='时间积分')
ck('W7','pi脉冲nV',h/(2*e*20e-9)*1e9,103.39)
Lp=np.sqrt(.02*.5e-9);C=2*np.pi*.4e-6
ck('W8','扩散长度um',Lp*1e6,3.162)
ck('W8','平均自由程nm',3*.02/1.4e6*1e9,42.86)
ck('W8','经验可见度',np.exp(-C/Lp),.4517)
ck('W8','弹道长度um',1.4e6*.5e-9*1e6,700)
E=ER(50e-9);Ip=E/Phi
ck('W9','ER微eV',E/e*1e6,15.240)
for n,ref in [(0,.9525),(-1,8.5725)]:
    ck('W9','能量微eV n='+str(n),E*(n+.25)**2/e*1e6,ref)
    ck('W9','电流与能量导数 n='+str(n),-fd(lambda u:(n+u)**2,.25)*Ip,-2*Ip*(n+.25),rtol=1e-7,atol=1e-20,method='有限差分')
ck('W9','速度m/s',.25*hb/(m*50e-9),578.84)
ck('W9','电流nA',-e*.25*hb/(m*50e-9)/(2*np.pi*50e-9)*1e9,-.2952)
T=10e-9;r=100e-9;dF=.2*Phi
ef=-dF/T/(2*np.pi*r)
ck('W10','感应场V/m',ef,-.13164)
torque=(-e)*r*ef
ck('W10','力矩时间积分/ħ',torque*T/hb,.2)
ck('W10','速度m/s',torque*T/(m*r),231.54)
powerwork=integ(lambda t:(-e)*ef*(torque*t/(m*r)),0,T)
ck('W10','电场功/动能',powerwork/(.5*m*(torque*T/(m*r))**2),1,method='功率积分')
phase=(-e)/hb/T*integ(lambda t:dF*t/T,0,T)
ck('W10','强制路径相位rad',phase,-.2*np.pi,method='时空作用积分')
mn=-9.662e-27
th=mn*.001*1e-6/hb
ck('W11','磁矩相位rad',th,-.09162);ck('W11','混合自旋可见度',abs((np.exp(1j*th)+np.exp(-1j*th))/2),.9958)
area=np.pi*(.5e-6)**2
ck('W12','h/e磁场周期mT',Phi/area*1e3,5.2657)
ck('W12','h/2e磁场周期mT',Phi/area/2*1e3,2.6328)
for w,ref in [(1,0),(2,-1)]:ck('W12','绕行差 '+str(w),np.cos(2*np.pi*w*.25),ref)
# Non-linking circle: parametrized A dot dr, normalized by flux.
ck('T1','偏心圆环流/Φ',integ(lambda t:((3+.5*np.cos(t))*.5*np.cos(t)+.25*np.sin(t)**2)/((3+.5*np.cos(t))**2+.25*np.sin(t)**2)/(2*np.pi),0,2*np.pi),0,method='偏心回路积分')
ck('T2','绕行数组合',2*.2-.35,.05)
ck('T2','相量',np.exp(-2j*np.pi*(2*.2-.35)),.95106-.30902j)
FL=integ(lambda r:2*np.pi*r*B*(1-(r/a)**2),0,a)
ck('T3','非均匀磁通Wb/1e-15',FL/1e-15,1.5708,method='径向面积积分')
ck('T3','外部A/1e-9',FL/(2*np.pi*200e-9)/1e-9,1.25)
ck('T3','相位rad',-e*FL/hb,-2.3865)
for x in [.1,.4,.8,1.2,2]:
    f=lambda s:s*(s/2-s**3/4 if s<1 else 1/(4*s))
    ck('T3','旋度/B0 r/a='+str(x),fd(f,x)/x,1-x*x if x<1 else 0,rtol=1e-6,method='有限差分旋度')
for n in [-1,0,1,2]:
    alpha=.4
    ratio=np.exp(1j*(n-alpha)*2*np.pi)
    ck('T4','切线连接 n='+str(n),ratio,np.exp(-2j*np.pi*alpha),method='本征态端点代入')
    z=np.linspace(0,2*np.pi,10001);psi=np.exp(1j*(n-alpha)*z)
    dz=z[1]-z[0];deriv2=(psi[2:]-2*psi[1:-1]+psi[:-2])/dz**2
    energy=np.mean((-deriv2)/psi[1:-1])
    ck('T4','扭转态能量 n='+str(n),energy,(n-alpha)**2,rtol=2e-6,method='有限差分本征方程')
K=100000*e;pr=np.sqrt(K*(K+2*m*c*c))/c;lr=h/pr
ck('T5','相对论波长pm',lr*1e12,3.70144)
ck('T5','条纹um',lr/1e-6*1e6,3.70144);ck('T5','位移um',lr/1e-6*.25*1e6,.92536)
ck('T5','非相对论误差百分比',(lam(100000)/lr-1)*100,4.778)
om=2*np.pi*1e6;V=1e-9;Tc=.25e-6
for t0 in [0,.1e-6,.25e-6,.6e-6]:
    numeric=e/hb*integ(lambda t:V*np.cos(om*t),t0,t0+Tc)
    exact=e*V/(hb*om)*(np.sin(om*(t0+Tc))-np.sin(om*t0))
    ck('T6','标势积分 t0='+str(t0),numeric,exact,rtol=1e-7,method='时间积分')
ck('T6','t0=0相位',e*V/(hb*om),.24180)
ck('T6','1ns抖动相位',e*V/hb*1e-9,.0015193)
ch1=np.array([1,0]);ch2=np.array([.5,np.sqrt(3)/2]);eta=(ch1+ch2)/np.sqrt(3)
ck('T7','环境重叠',np.vdot(ch2,ch1),.5,method='态内积')
ck('T7','投影成功率',.5*(abs(eta@ch1)**2+abs(eta@ch2)**2),.75,method='矩阵投影')
for delta in [0,np.pi/2,np.pi]:
    vector=np.exp(1j*delta)*ch1+ch2
    ck('T7','求迹强度 δ='+str(delta),np.vdot(vector,vector),2+np.cos(delta),method='联合态范数')
    ck('T7','条件强度 δ='+str(delta),abs(eta@vector)**2,.75*(2+2*np.cos(delta)),method='联合态投影')
for val,ref in [(.25,0),(.5,-1),(1,1)]:
    ck('T8','二点噪声平均 a='+str(val),np.mean(np.exp(2j*np.pi*np.array([-val,val]))),ref,method='离散平均')
ck('T8','同方差高斯误用结果',np.exp(-2*np.pi**2*.25**2),.2912)
da=.05*area;B1=Phi/(2*da)
ck('T9','面积包络零点mT',B1*1e3,52.657)
for b in [0,.2*B1,B1,1.5*B1]:
    q=integ(lambda A:np.exp(-2j*np.pi*b*A/Phi),area-da,area+da)/(2*da)
    exact=np.exp(-2j*np.pi*b*area/Phi)*np.sinc(2*b*da/Phi)
    ck('T9','面积相量 B/B1='+str(b/B1),q,exact,rtol=1e-6,method='面积分布积分')
def leakage(K,bl,ell,L):
    # Exact circular arc in constant B, followed by straight drift to x screen.
    sp=v(K);w=e*bl/m;theta=np.arcsin(w*ell/sp)
    yarc=sp/w*(2*np.sin(theta/2)**2)
    return yarc+L*np.tan(theta)
yd=leakage(100,1e-7,.001,1)
ck('T10','精确轨道偏转um',yd*1e6,2.9670,method='圆弧与自由漂移')
ck('T10','偏转/条纹',yd/(lam(100)/1e-6),.024192,method='精确轨道')
ck('T10','漏场上限nT',.01*Phi/(1e-6*.001*(1+.001/2))*1e9,41.336)
for kval in [30,100,300]:
    ck('T10','归一化能量控制 K='+str(kval),leakage(kval,1e-7,.001,1)/(lam(kval)/1e-6),.024192,method='精确轨道与波长')
sigK=5*e;dl=.5e-9;coh=hb*v(100)/sigK;s=dl/coh
ck('T11','相干长度nm',coh*1e9,.78077)
ck('T11','相位标准差',s,.64040);ck('T11','线性高斯积分',gaussian_phase(s),.81460,method='能量分布数值积分')
ck('T11','二阶相位参数',s*.05/2,.01601)
k0=m*v(100)/hb
nonlinear=integ(lambda z:np.exp(-z*z/2)/np.sqrt(2*np.pi)*np.exp(1j*k0*dl*(np.sqrt(1+.05*z)-1)),-9,9)
ck('T11','精确能量积分幅度与近似',abs(nonlinear),.81460,rtol=.002,method='未展开波数数值积分')
E=ER(50e-9)
ck('T12','固定分支做功微eV',(E*.8**2-E*.2**2)/e*1e6,9.1440)
ck('T12','功积分/ER',integ(lambda u:2*u,.2,.8),.6,method='磁通积分')
ck('T12','初电流nA',-2*E/Phi*.2*1e9,-.2362);ck('T12','末电流nA',-2*E/Phi*.8*1e9,-.9446)
for alpha in [.35,.45,.5,.55,.65]:
    mat=np.array([[alpha*alpha,.05],[.05,(1-alpha)**2]])
    evals,vec=np.linalg.eigh(mat);x=alpha-.5
    ref=np.array([.25+x*x-np.sqrt(x*x+.05**2),.25+x*x+np.sqrt(x*x+.05**2)])
    ck('T13','矩阵本征值 α='+str(alpha),evals,ref,method='独立矩阵对角化')
    dH=np.diag([2*alpha,2*(alpha-1)])
    derivative=np.vdot(vec[:,0],dH@vec[:,0])
    ck('T13','下支电流导数 α='+str(alpha),derivative,2*x-x/np.sqrt(x*x+.05**2),method='本征态Hellmann-Feynman矩阵元')
ck('T13','中心隙/ER',np.linalg.eigvalsh(np.array([[.25,.05],[.05,.25]]))[1]-np.linalg.eigvalsh(np.array([[.25,.05],[.05,.25]]))[0],.1,method='矩阵对角化')
fs=np.arange(8)/8;g=1+.08*np.cos(2*np.pi*fs)+.12*np.cos(4*np.pi*fs)
for j,ref in enumerate([1.2,1.05657,.88,.94343,1.04,.94343,.88,1.05657]):
    ck('T14','采样j='+str(j),g[j],ref)
ck('T14','离散系数',[np.mean(g),2*np.mean(g*np.cos(2*np.pi*fs)),2*np.mean(g*np.cos(4*np.pi*fs))],[1,.08,.12],method='离散傅里叶求和')
for n in range(5):
    phase=-e*(n*h/(2*e))/hb
    ck('T15','电子相量 N='+str(n),np.exp(1j*phase),(-1)**n)
    ck('T15','配对相量 N='+str(n),np.exp(2j*phase),1)
for f1,f2 in [(0,0),(1/3,1/3),(.1,.7),(.5,.2),(1.1,.7)]:
    direct=abs(1+np.exp(-2j*np.pi*f1)+np.exp(-2j*np.pi*(f1+f2)))**2/9
    expr=(3+2*np.cos(2*np.pi*f1)+2*np.cos(2*np.pi*f2)+2*np.cos(2*np.pi*(f1+f2)))/9
    ck('T16','三相量透射 '+str((f1,f2)),direct,expr,method='复数振幅相加')
asrc=np.pi*(80e-9)**2;df=lam(150)*.6/.8e-6
ck('T17','内部场周期T',Phi/asrc,.20569)
ck('T17','半周期内部场T',Phi/(2*asrc),.10285)
ck('T17','条纹um',df*1e6,75.1029);ck('T17','半条纹位移um',df/2*1e6,37.5515)
V0=1.4/1.49;sigtheta=3*e*.5e-9/(hb*v(150));GK=float(gaussian_phase(sigtheta).real);GF=float(gaussian_phase(2*np.pi*.04).real)
ck('T17','振幅可见度',V0,.939597)
ck('T17','相干长度nm',hb*v(150)/(3*e)*1e9,1.59374)
ck('T17','相位宽度',sigtheta,.31373);ck('T17','能量积分因子',GK,.951978,method='高斯数值积分')
ck('T17','磁通积分因子',GF,.968911,method='高斯数值积分')
ck('T17','总可见度',V0*GK*GF,.866668)
sn=math.sqrt(-2*math.log(.85/(V0*GK)))/(2*np.pi)
ck('T17','噪声上限比',sn,.050831);ck('T17','噪声Wb/1e-16',sn*Phi/1e-16,2.1023)
ck('T17','漏场上限nT',.01*Phi/(.8e-6*.002*(1+.002/(2*.6)))*1e9,25.805)
E=ER(100e-9);ns=np.arange(-6,7)
for u in [.01,.25,.5,.75,.99,1.25]:
    order=np.argsort((ns+u)**2);occ=ns[order[:2]]
    energy=np.sum((occ+u)**2);frac=u%1
    ck('T18','泡利占据能量 u='+str(u),energy,2*frac*frac-2*frac+1,method='整数能级排序')
    current=-2*np.sum(occ+u)
    ck('T18','占据电流 u='+str(u),current,2-4*frac,method='两个占据态电流求和')
ck('T18','ER微eV',E/e*1e6,3.80998)
ck('T18','平衡能量微eV',.625*E/e*1e6,2.38124)
ck('T18','平衡电流nA',E/Phi*1e9,.147600)
ck('T18','孤立末能量微eV',1.625*E/e*1e6,6.19122)
ck('T18','孤立末电流nA',-3*E/Phi*1e9,-.442801)
ck('T18','做功积分/ER',integ(lambda u:4*u-2,.25,1.25),1,method='磁通做功积分')
ck('T18','两个简并态混合电流',(-2*np.sum([0,-1])-2*np.sum([0,1]))/2,0,method='简并态等量混合')
out=dict(count=len(records),count_description='一项为一个参数实例的一次数值比较；复数或向量以最大分量误差计作一次。覆盖12例题、18习题与规范理论，积分和差分网格点不计数，次数不等于独立公式数量。',
 all_passed=all(x['passed'] for x in records),groups=sorted(set(x['group'] for x in records)),records=records)
(R/'检查/第二十八讲_公式核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='records'},ensure_ascii=False))
for x in records:
    if not x['passed']:print(json.dumps(x,ensure_ascii=False))
assert out['all_passed']
