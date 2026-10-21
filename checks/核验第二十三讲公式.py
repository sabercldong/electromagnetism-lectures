from pathlib import Path
import math,json,numpy as np
records=[]
def check(group,name,a,b,tol=2e-7):
    err=float(np.max(np.abs(np.asarray(a)-b)))/max(1.,float(np.max(np.abs(b))))
    records.append(dict(group=group,name=name,normalized_error=err,tolerance=tol,passed=err<=tol))
def nodes(edges,fixed,injection=None):
    names=sorted(set(x for a,b,z in edges for x in (a,b)))
    free=[x for x in names if x not in fixed];idx={x:k for k,x in enumerate(free)}
    A=np.zeros((len(free),len(free)),complex);b=np.zeros(len(free),complex)
    for a,c,z in edges:
        for u,v in [(a,c),(c,a)]:
            if u in idx:
                A[idx[u],idx[u]]+=1/z
                if v in idx:A[idx[u],idx[v]]-=1/z
                else:b[idx[u]]+=fixed[v]/z
    for u,i in (injection or {}).items():b[idx[u]]+=i
    ans=dict(fixed);ans.update(zip(free,np.linalg.solve(A,b)));return ans
def derivative(f,x,h=1e-5):
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
g,w=np.polynomial.legendre.leggauss(160)
def integ(f,a,b):return (b-a)/2*np.dot(w,f((b-a)*g/2+(a+b)/2))
for Q in [.5,2,10,50]:
    R=1/Q
    for x in [.2,.8,1,1.2,3]:
        V=nodes([('s','a',R),('a','b',1j*x),('b','g',1/(1j*x))],{'s':1,'g':0})
        I=(1-V['a'])/R
        check('独立节点','串联电流',I,1/(R+1j*(x-1/x)))
for Q in [.5,1,2,10,50]:
    R=1/Q;roots=[]
    for sign in [-1,1]:
        root=max(np.roots([1,sign*R,-1]));roots.append(root)
        target=math.hypot(1,R/2)-sign*R/2
        check('独立求根','半功率边界',root,target)
    check('独立求根','带宽与几何中心',[max(roots)-min(roots),np.prod(roots)],[R,1])
for Q in [1,2,5,50]:
    R=1/Q
    for part in ['C','L']:
        xp=math.sqrt(1-1/(2*Q*Q));xp=xp if part=='C' else 1/xp
        def amp(x):
            V=nodes([('s','a',R),('a','b',1j*x),('b','g',1/(1j*x))],{'s':1,'g':0})
            return abs(V['b'] if part=='C' else V['a']-V['b'])
        peak=Q/math.sqrt(1-1/(4*Q*Q))
        check('元件电压','节点峰值',amp(xp),peak)
        f=lambda x:amp(x)**2/peak**2
        check('有限差分','元件电压峰驻点',derivative(f,xp,1e-4/(Q+1)),0,tol=3e-6)
for x in [.5,1,2]:
    t=np.linspace(0,2*np.pi,10001);i=np.sqrt(2)*np.cos(t);uc=np.sqrt(2)/x*np.sin(t)
    W=.5*i*i+.5*uc*uc
    check('独立积分','储能均值',integ(lambda t:np.cos(t)**2+np.sin(t)**2/x**2,0,2*np.pi)/(2*np.pi),.5*(1+1/x**2))
    check('储能极值','偏离谐振',max(W),max(1,1/x**2))
def rk4(rhs,y,T,steps=2500):
    y=np.array(y,float);h=T/steps
    for k in range(steps):
        a=rhs(y);b=rhs(y+h*a/2);c=rhs(y+h*b/2);d=rhs(y+h*c);y+=h*(a+2*b+2*c+d)/6
    return y
for Q in [.6,1,2,10,50]:
    R=1/Q;alpha=R/2;wd=math.sqrt(1-alpha*alpha);T=2*np.pi/wd
    rhs=lambda y:np.array([y[1],-R*y[1]-y[0],R*y[1]**2])
    end=rk4(rhs,[1,0,0],T)
    check('独立微分方程','同相峰衰减',end[:2],[math.exp(-alpha*T),0],tol=2e-7)
    check('能量积分','自由衰减损耗',end[2],.5-.5*np.dot(end[:2],end[:2]),tol=2e-7)
def lossyZ(a,x):
    r=math.sqrt(a)
    v=nodes([('t','m',1j*x),('m','g',r),('t','g',1/(1j*x))],{'g':0},{'t':1})
    return v['t']
for a in [.01,.2,.9,1.5,2.2,2.6]:
    for x in [.2,1,2]:
        r=math.sqrt(a)
        check('独立节点','有损并联端口',lossyZ(a,x),(r+1j*x)/(1-x*x+1j*x*r))
    if a<1:
        xp=math.sqrt(1-a);z=lossyZ(a,xp)
        check('相位条件','并联零虚部',[z.real,z.imag],[1/math.sqrt(a),0])
    if a<1+math.sqrt(2):
        xp=math.sqrt(math.sqrt(1+2*a)-a);f=lambda x:abs(lossyZ(a,x))**2/abs(lossyZ(a,xp))**2
        check('有限差分','并联阻抗峰',derivative(f,xp),0,tol=2e-6)
for R,L,C in [(2000,.05,20e-6),(1000,.01,1e-6),(10,2,.2)]:
    omega=1/math.sqrt(L*C);I=.01
    v=nodes([('t','g',R),('t','g',1j*omega*L),('t','g',1/(1j*omega*C))],{'g':0},{'t':I})['t']
    check('独立节点','纯并联谐振',v,I*R)
    t=np.linspace(0,2*np.pi,10001);iL=np.sqrt(2)*abs(v)/(omega*L)*np.sin(t);u=np.sqrt(2)*abs(v)*np.cos(t)
    energy=.5*L*iL*iL+.5*C*u*u
    check('能量交换','并联恒定储能',np.max(abs(energy-C*abs(v)**2)),0)
for E,rs,n,zl in [(12,3,.25,48),(10,2,3,4+2j),(5,1,.5,2-1j)]:
    # Unknowns are U1,U2,I1,IL; four port/source equations are solved together.
    A=np.array([[1,0,rs,0],[1,-n,0,0],[0,0,n,-1],[0,1,0,-zl]],complex)
    u1,u2,i1,il=np.linalg.solve(A,[E,0,0,0])
    check('独立端口方程','阻抗折算',u1/i1,n*n*zl)
    check('复功率','理想变压器',u1*i1.conjugate(),u2*il.conjugate())
for z1,z2,zl,rc,xm in [(2+4j,1+1j,100,10000,2000),(.1+.2j,.2+.3j,10+4j,1000,500),(1+2j,2+1j,20-3j,2000,1000)]:
    E=230
    v=nodes([('s','m',z1),('m','g',rc),('m','g',1j*xm),('m','l',z2),('l','g',zl)],{'s':E,'g':0})
    zp=1/(1/rc+1/(1j*xm)+1/(z2+zl));i1=E/(z1+zp);um=E-z1*i1;il=um/(z2+zl)
    check('独立节点','完整T形模型',[v['m'],v['l']],[um,il*zl])
    pin=(E*i1.conjugate()).real
    check('有功守恒','变压器全部损耗',pin,abs(i1)**2*z1.real+abs(il)**2*(z2.real+zl.real)+abs(um)**2/rc)
Ephase=100*np.exp(1j*np.array([0,-2*np.pi/3,2*np.pi/3]))
for impedances,rline,rn in [([10,20,40],0,2),([10,20,40],1,2),([10+5j,20-3j,40+2j],.5,4),([10,20,40],0,None)]:
    fixed={'a':Ephase[0],'b':Ephase[1],'c':Ephase[2],'g':0};edges=[]
    for k,z in zip('abc',impedances):
        if rline:edges.extend([(k,k+'x',rline),(k+'x','N',z)])
        else:edges.append((k,'N',z))
    if rn is not None:edges.append(('N','g',rn))
    v=nodes(edges,fixed);ys=1/(np.array(impedances)+rline);gn=0 if rn is None else 1/rn
    vn=np.dot(ys,Ephase)/(sum(ys)+gn);i=ys*(Ephase-vn);ul=np.array(impedances)*i
    check('独立多节点','中性点位移',v['N'],vn)
    pin=np.dot(Ephase,i.conjugate()).real
    check('有功守恒','三相及导线',pin,np.dot(abs(i)**2,np.real(impedances))+rline*sum(abs(i)**2)+gn*abs(vn)**2)
for q in [2,10,20]:
    G=.01;B=G*q
    for x in [.4,1,2]:
        for gn in [0,.2]:
            ys=np.array([G+1j*B*x,G,G-1j*B/x])
            edges=[(k,'N',1/y) for k,y in zip('abc',ys)]
            if gn:edges.append(('N','g',1/gn))
            v=nodes(edges,{'a':Ephase[0],'b':Ephase[1],'c':Ephase[2],'g':0})['N']
            sx=x*x+1/(x*x);aa=(3*G+gn)/G
            expected=q*q*(sx+1)/(aa*aa+q*q*(sx-2))
            check('独立节点','浮置反应负载幅值',abs(v)**2/100**2,expected)
            currents=ys*(Ephase-v);pin=np.dot(Ephase,currents.conjugate()).real
            check('有功守恒','反应负载与中性线',pin,G*sum(abs(Ephase-v)**2)+gn*abs(v)**2)
for q,gn in [(10,0),(2,0),(10,.2)]:
    aa=3+gn/.01
    f=lambda x:q*q*(x*x+1/x**2+1)/(aa*aa+q*q*(x-1/x)**2)
    second=(f(1+1e-4)-2*f(1)+f(1-1e-4))/1e-8
    target=8*q*q*(aa*aa-3*q*q)/aa**4
    check('有限差分','中性点峰谷判据',second,target,tol=2e-6)
for R0,E,Z0,limit,qmin,RL in [(5,10,100,80,6,30),(2,12,50,120,2,20),(4,8,80,100,3,10)]:
    lower=max(R0,E*Z0/limit);upper=Z0/qmin
    lo=max(0,lower-R0);hi=upper-R0
    assert hi>=lo
    opt=np.clip(R0,lo,hi);grid=np.linspace(lo,hi,100001)
    maximum=max(E*E*grid/(R0+grid)**2)
    check('约束优化','谐振匝数匹配',maximum,E*E*opt/(R0+opt)**2,tol=2e-7)
for phi in [math.acos(.8),-math.acos(.8),0]:
    I=2;R=1;X=3;V0=230;parallel=I*(R*math.cos(phi)+X*math.sin(phi));perp=I*(X*math.cos(phi)-R*math.sin(phi))
    V=math.sqrt(V0**2-perp**2)-parallel;current=I*np.exp(-1j*phi);source=V+(R+1j*X)*current
    check('复电压方程','超前滞后端电压',abs(source),V0)
    check('有功守恒','电压调整率',np.real(source*current.conjugate()),V*I*math.cos(phi)+I*I*R)
for phi in [-.8,0,.6,1.2]:
    currents=2*np.exp(1j*(np.array([0,-2*np.pi/3,2*np.pi/3])-phi))
    wa=((Ephase[0]-Ephase[1])*currents[0].conjugate()).real;wc=((Ephase[2]-Ephase[1])*currents[2].conjugate()).real
    total=np.dot(Ephase,currents.conjugate())
    check('两瓦特表','平衡有功及无功',[wa+wc,np.sqrt(3)*(wc-wa)],[total.real,total.imag])
    # Time-domain three-phase power is checked at non-special phases.
    for t in [.23,1.1,2.2]:
        p=sum(np.sqrt(2)*np.real(Ephase*np.exp(1j*t))*np.sqrt(2)*np.real(currents*np.exp(1j*t)))
        check('瞬时合成','平衡三相功率',p,total.real)
for loads in [[10,20,40],[10+3j,20-4j,5+2j]]:
    voltages=Ephase-np.roll(Ephase,-1);branch=voltages/loads
    line=branch-np.roll(branch,1);s=np.dot(voltages,branch.conjugate());wa=((Ephase[0]-Ephase[1])*line[0].conjugate()).real;wc=((Ephase[2]-Ephase[1])*line[2].conjugate()).real
    check('独立支路','三角形复功率',np.dot(Ephase,line.conjugate()),s)
    check('两瓦特表','不平衡代数和',wa+wc,s.real)
angles=np.array([0,2*np.pi/3,4*np.pi/3]);axes=np.array([np.cos(angles),np.sin(angles)])
for t in [.1,.7,1.5,2.3,3.1]:
    vec=axes@np.cos(t-angles)
    check('空间合成','旋转磁场',vec,1.5*np.array([math.cos(t),math.sin(t)]))
    check('空间合成','三次同相谐波',axes@np.full(3,math.cos(3*t)),[0,0])
for Pc,Pcu,kmax in [(50,200,1),(20,100,1),(200,100,1)]:
    opt=min(kmax,math.sqrt(Pc/Pcu));f=lambda k:800*k/(800*k+Pc+Pcu*k*k)
    grid=np.linspace(0,kmax,100001)
    check('效率优化','铜损铁损平衡',max(f(grid)),f(opt),tol=2e-7)
# Numerical values printed in the handout are compared at their stated precision.
alpha=math.log(2)/.005;wd=2*np.pi/.001;omega=math.hypot(wd,alpha)
numeric=[
 ('W2电容微法',1/((2*np.pi*math.sqrt(990000))**2*(20/(400*np.pi)))*1e6,1.60763),
 ('W5相位频率',1000*math.sqrt(.99),994.987),
 ('W5阻抗峰频率',1000*math.sqrt(math.sqrt(1.02)-.01),999.975),
 ('W9电压',abs(230/(102+4j))*10,22.5317),
 ('W10空载电流',math.hypot(.023,.115),.117277),
 ('W10磁通',230/(math.sqrt(2)*np.pi*50*500*.002),1.03536),
 ('T1失谐',1/math.sqrt(1+2500*(1.01-1/1.01)**2),.708859),
 ('T3Q',omega/(2*alpha),22.6673),
 ('T3电容微法',1/(.01*omega**2)*1e6,2.53180),
 ('T13中性电阻损耗',abs((250-50j*np.sqrt(3))/27)**2/2,48.01097),
 ('T18中性线电阻界',1/(np.sqrt(3)*.1-.03),6.98299)
]
for name,a,b in numeric:check('数字代入',name,a,b,tol=5e-6)
groups={}
for r in records:
    d=groups.setdefault(r['group'],dict(count=0,max_error=0));d['count']+=1;d['max_error']=max(d['max_error'],r['normalized_error'])
report=dict(count=len(records),count_description='独立比较的参数实例数；不是独立公式数量',all_passed=all(r['passed'] for r in records),groups=groups,records=records)
p=Path(__file__).resolve().parent/'检查/第二十三讲_公式核验.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False))
assert report['all_passed'],[r for r in records if not r['passed']]
