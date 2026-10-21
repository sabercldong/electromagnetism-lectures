from pathlib import Path
import numpy as np,math,json
records=[]
def check(group,name,a,b,tol=2e-7):
    e=float(np.max(np.abs(np.asarray(a)-b)))/max(1.,float(np.max(np.abs(b))))
    records.append(dict(group=group,name=name,normalized_error=e,tolerance=tol,passed=e<=tol))
def network(edges,fixed):
    nodes=sorted(set(v for a,b,z in edges for v in (a,b)));free=[v for v in nodes if v not in fixed];ix={v:k for k,v in enumerate(free)};A=np.zeros((len(free),len(free)),complex);b=np.zeros(len(free),complex)
    for a,c,z in edges:
        for u,v in [(a,c),(c,a)]:
            if u in ix:
                A[ix[u],ix[u]]+=1/z
                if v in ix:A[ix[u],ix[v]]-=1/z
                else:b[ix[u]]+=fixed[v]/z
    V=dict(fixed);V.update(zip(free,np.linalg.solve(A,b)));return V
def bridge(zs,zg,U=1):
    z1,z2,z3,z4=zs;V=network([('A','C',z1),('A','D',z2),('C','B',z3),('D','B',z4),('C','D',zg)],{'A':U,'B':0});return (V['C']-V['D'])/zg,V
for n in [1,2,3]:
    for a in [.01,.2,1,math.sqrt(6),5,20]:
        edges=[]
        for k in range(1,n+1):edges.extend([(str(k-1),str(k),1),(str(k),'g',1/(1j*a))])
        v=network(edges,{'0':1,'g':0})[str(n)]
        d=[None,1+1j*a,1+3j*a-a*a,1+6j*a-5*a*a-1j*a**3][n]
        check('节点方程','RC级联',v,1/d)
for rt,rl,c in [(1500,3000,1e-7),(2,5,.3),(10,1,.2)]:
    for omega in [.1,10,10000]:
        v=network([('A','X',rt),('X','B',rl),('X','B',1/(1j*omega*c))],{'A':1,'B':0})['X'];K=rl/(rt+rl);tau=rt*rl/(rt+rl)*c
        check('节点方程','加载滤波',v,K/(1+1j*omega*tau))
for z1,z2,z3,z4,zg in [(2+3j,4-2j,3,5,2),(100-1000j,1000,1000,1000,2000),(4,3,6,8,1)]:
    ig,V=bridge([z1,z2,z3,z4],zg);uth=z3/(z1+z3)-z4/(z2+z4);zth=z1*z3/(z1+z3)+z2*z4/(z2+z4)
    check('节点方程','有限检测器',ig,uth/(zth+zg))
for w in [100,1000,10000]:
    c2=1e-6;r3=2000;r4=3000;c4=1e-8;cx=c2*r4/r3;rx=r3*c4/c2
    ig,V=bridge([rx+1/(1j*w*cx),1/(1j*w*c2),r3,r4/(1+1j*w*r4*c4)],1000)
    check('电桥平衡','电容桥',ig,0)
    r2=1000;r3=2000;r4=20000;c4=5e-8;rx=r2*r3/r4;lx=r2*r3*c4
    ig,V=bridge([rx+1j*w*lx,r2,r3,r4/(1+1j*w*r4*c4)],1000);check('电桥平衡','麦克斯韦桥',ig,0)
for r1,r3,c1,c3 in [(1,1,1,1),(2,1,1,3),(4,3,.2,.5)]:
    w=1/math.sqrt(r1*r3*c1*c3);ratio=r1/r3+c3/c1
    ig,V=bridge([r1+1/(1j*w*c1),ratio,r3/(1+1j*w*r3*c3),1],2)
    check('电桥平衡','维恩桥',ig,0)
for R,rg,dr in [(1,1,.01),(1000,2000,40),(3,.5,-.2)]:
    ig,V=bridge([R+dr,R,R,R],rg,U=12);check('节点方程','电阻灵敏度精确',ig,-12*dr/(4*R*(R+rg)+(3*R+2*rg)*dr))
for z in [.1-1j,2-.5j,1-.1j]:
    R=1000;rg=2000;ig,V=bridge([z*R,R,R,R],rg);h=ig
    recovered=(R-h*(R*R+2*R*rg))/(1+h*(3*R+2*rg));check('独立反演','未知桥臂',recovered,z*R)
for r1,x1,a,b,m in [(1,4,2,3,2),(2,5,3,-1,1),(.1,1,.5,0,.3)]:
    I=np.linalg.solve([[r1+1j*x1,1j*m],[1j*m,a+1j*b]],[10,0]);zin=r1+1j*x1+m*m/(a+1j*b)
    check('互感方程','反射阻抗',10/I[0],zin);check('互感能量','有功',np.real(10*I[0].conjugate()),r1*abs(I[0])**2+a*abs(I[1])**2)
for l1,l2,m in [(2,3,.4),(1,1,.99),(4,9,5.9)]:
    deriv=np.linalg.solve([[l1,m],[m,l2]],[1,1]);le=(l1*l2-m*m)/(l1+l2-2*m)
    check('互感方程','并联等效',sum(deriv),1/le)
    eig=np.linalg.eigvalsh([[l1,m],[m,l2]]);check('磁能','非负矩阵',max(0,-min(eig)),0)
g,w=np.polynomial.legendre.leggauss(180)
def integral(f,a,b):return (b-a)/2*np.dot(w,f((b-a)*g/2+(a+b)/2))
for U,I,pu,pi in [(100,10,0,-math.atan2(8,6)),(12,3,.5,-.7),(5,4,-.2,1.8)]:
    check('独立积分','瞬时平均功率',integral(lambda t:2*U*I*np.cos(t+pu)*np.cos(t+pi),0,2*np.pi)/(2*np.pi),U*I*math.cos(pu-pi))
check('独立积分','畸变有功',integral(lambda t:220*math.sqrt(2)*np.cos(t)*(10*math.sqrt(2)*np.cos(t-math.acos(.8))+6*math.sqrt(2)*np.cos(3*t)),0,2*np.pi)/(2*np.pi),1760)
for r,g,b in [(5,.02,-.04),(1,.1,.2),(2,.03,0)]:
    Y=g+1j*b;V=230/(1+r*Y);I=Y*V;D=(1+r*g)**2+(r*b)**2
    check('线路能量','有功守恒',np.real(230*I.conjugate()),g*abs(V)**2+r*abs(I)**2)
    f=lambda q:230**2*(g*g+q*q)/((1+r*g)**2+(r*q)**2);h=1e-5
    der=(f(b-2*h)-8*f(b-h)+8*f(b+h)-f(b+2*h))/(12*h)
    check('有限差分','线路电流极值',der,2*b*230**2*(1+2*r*g)/D**2,tol=2e-6)
for rs,xs in [(3,4),(2,-3),(.5,0)]:
    for R in [.5*rs,rs,2*rs,math.hypot(rs,xs)]:
        f=lambda q:144*q/((rs+q)**2+xs*xs);h=1e-4*rs
        der=(f(R-2*h)-8*f(R-h)+8*f(R+h)-f(R+2*h))/(12*h)
        check('有限差分','纯电阻匹配',der,144*(rs*rs+xs*xs-R*R)/((rs+R)**2+xs*xs)**2)
check('数字代入','W1纹波',100/math.sqrt(1+4*math.pi**2),15.718,tol=3e-5)
check('数字代入','W4微瓦',(.003*500)**2/(1+(math.pi*1000*500*30e-6)**2)/500*1e6,2.026,tol=3e-4)
check('数字代入','W11电容微法',1000*(4/3-math.sqrt(1-.9**2)/.9)/(100*math.pi*220**2)*1e6,55.84,tol=1e-4)
check('数字代入','T12视在',math.hypot(2592,1152),2836.47,tol=1e-6)
ba=math.sqrt((230/208)**2-1.1**2)/5
for sign in [-1,1]:
    C=(.04+sign*ba)/(100*math.pi);Y=.02+1j*(100*math.pi*C-.04);V=230/(1+5*Y);I=Y*V
    check('耐压约束','T18边界',abs(V),208);check('数字代入','T18最小电流',abs(I),6.2712,tol=1e-6)
for d,k,x in [(1,.75,.2),(2,.5,.6),(3,.9,1.2)]:
    c0=1/d;c1=1/(d-k*x)
    h=c1/(c0+c1)-.5
    check('传感器反演','介质厚度',4*d*h/(k*(1+2*h)),x)
    step=1e-5*d;f=lambda y:(1/(d-k*y))/(1/d+1/(d-k*y))-.5
    derivative=(f(x+step)-f(x-step))/(2*step)
    check('有限差分','厚度灵敏度',derivative,k*d/(2*d-k*x)**2)
for r,w,L,a,x in [(1,1,1,.3,.2),(2,10,.5,.1,-.3),(4,2,3,.4,.5)]:
    z1=r+1j*w*(L+a*x);z3=r+1j*w*(L-a*x)
    h=z3/(z1+z3)-.5
    check('传感器响应','差动电感',h,-1j*w*a*x/(2*(r+1j*w*L)))
for R,theta,I,beta in [(100,20,.01,.004),(200,10,.01,.003),(50,40,.005,.005)]:
    T=theta*I*I*R/(1-theta*I*I*R*beta)
    check('热收支','恒流稳态',T,theta*I*I*R*(1+beta*T))
    step=1e-5;P=lambda dr:12**2*(R+dr)/(2*R+dr)**2
    check('有限差分','匹配桥臂热功率',(P(step)-P(-step))/(2*step),0)
for a in [.01,.05,.2,1]:
    H=1j*a/(1+1j*a);Hd=1j*a
    check('近似误差','高通复误差',abs((Hd-H)/H),a)
    check('近似误差','高通幅度误差',abs(Hd)/abs(H)-1,math.sqrt(1+a*a)-1)
for rs,eta in [(2,.8),(1,.3),(5,.9)]:
    R=max(rs,rs*eta/(1-eta));E=10
    grid=np.linspace(rs*eta/(1-eta),max(100*rs,R*10),200001)
    P=E*E*grid/(rs+grid)**2
    check('约束优化','效率界内最大功率',max(P),E*E*R/(rs+R)**2,tol=2e-6)
groups={}
for q in records:
    g=groups.setdefault(q['group'],dict(count=0,max_error=0));g['count']+=1;g['max_error']=max(g['max_error'],q['normalized_error'])
report=dict(count=len(records),count_description='独立比较的参数实例数；不是独立公式数量',all_passed=all(q['passed'] for q in records),groups=groups,records=records)
p=Path(__file__).resolve().parent/'检查/第二十二讲_公式核验.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False));assert report['all_passed'],[q for q in records if not q['passed']]
