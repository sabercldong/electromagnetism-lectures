from pathlib import Path
import numpy as np,json,math
records=[]
def check(group,name,a,b,tol=2e-7,scale=None):
    error=abs(float(a)-float(b))/(scale if scale is not None else max(abs(float(b)),1.0))
    records.append(dict(group=group,name=name,actual=float(a),expected=float(b),normalized_error=error,tolerance=tol,passed=error<=tol))
def network(edges,fixed,injection=None):
    nodes=sorted(set(x for a,b,r in edges for x in (a,b)));free=[a for a in nodes if a not in fixed];idx={a:i for i,a in enumerate(free)};G=np.zeros((len(free),len(free)));J=np.zeros(len(free));injection=injection or {}
    for a in free:J[idx[a]]=injection.get(a,0)
    for a,b,r in edges:
        for u,v in [(a,b),(b,a)]:
            if u in idx:
                i=idx[u];G[i,i]+=1/r
                if v in idx:G[i,idx[v]]-=1/r
                else:J[i]+=fixed[v]/r
    sol=np.linalg.solve(G,J);V=dict(fixed);V.update({a:float(sol[idx[a]]) for a in free});I=[(V[a]-V[b])/r for a,b,r in edges];return V,I
def eqres(edges,A,B):
    V,I=network(edges,{B:0},{A:1});return V[A],I
x,w=np.polynomial.legendre.leggauss(180)
def integral(f,a,b):
    q=(b-a)*x/2+(a+b)/2;return (b-a)/2*np.dot(w,f(q))
for E,r in [(3,.5),(12,2),(13.8,.6)]:
    for R in [.2*r,r,4*r,20*r]:
        I=E/(R+r);check('源及负载','能量收支',E*I,I*I*(R+r))
        h=R*1e-4;f=lambda z:E*E*z/(z+r)**2
        check('有限差分','负载功率导数',(f(R-2*h)-8*f(R-h)+8*f(R+h)-f(R+2*h))/(12*h),E*E*(r-R)/(R+r)**3,tol=2e-7)
check('数字代入','W1储能与热',15*2,12*2+4*.4+4*1.1)
check('数字代入','W2约束功率',12**2*8/(8+2)**2,11.52)
f=lambda a,b:(b*10-a*4)/(a-b)
for ix,exact in [(0,-6),(1,12)]:
    t=[2.,1.];h=1e-5;t[ix]+=h;up=f(*t);t[ix]-=2*h;down=f(*t);check('有限差分','W3内阻反演',(up-down)/(2*h),exact)
check('积分','W4温差电动势',integral(lambda T:40e-6+.04e-6*(T-300),300,500),.0088,scale=.0088)
check('数字代入','W4负载微瓦',(.0088/100)**2*20*1e6,.15488)
I=2.8;Qc=.02*280*I-.5*2*I*I-.2*20;Pin=.02*20*I+2*I*I;Qh=.02*300*I+.5*2*I*I-.2*20
check('热平衡','W5冷热端能量',Qc+Pin,Qh);check('数字代入','W5制冷量',Qc,3.84)
ne=8.5e28;e=1.6022e-19;m=9.1094e-31;rho=1.7e-8;j=2.4e6;tau=m/(ne*e*e*rho)
check('数字代入','W6漂移速度',j/(ne*e),1.7623e-4,tol=2e-5,scale=1.7623e-4)
check('微观输运','Drude一致性',e*(rho*j)*tau/m,j/(ne*e),scale=j/(ne*e))
for ell in [.3,1,4]:
    check('积分','W7变截面电阻',integral(lambda z:1/(1+z/ell)**2,0,ell),ell/2)
    check('积分','W7前半段发热比例',integral(lambda z:1/(1+z/ell)**2,0,ell/2)/(ell/2),2/3)
for vals in [(2,4,3,3),(1,5,2,7),(4,1,3,8)]:
    r1,r2,r3,r4=vals
    for rg in [.2,2,10]:
        edges=[('A','C',r1),('C','B',r2),('A','D',r3),('D','B',r4),('C','D',rg)]
        V,I=network(edges,{'A':12,'B':0});vth=12*(r2/(r1+r2)-r4/(r3+r4));rt=r1*r2/(r1+r2)+r3*r4/(r3+r4)
        check('网络方程','非平衡桥检测电流',(V['C']-V['D'])/rg,vth/(rt+rg))
        check('能量收支','电桥全部耗散',12*(I[0]+I[2]),sum(i*i*r for i,(a,b,r) in zip(I,edges)))
check('数字代入','W9双源负载',(3/10+1/20)/(1/10+1/20+1/10),1.4)
cube=[(a,b,1.) for a in range(8) for b in range(a+1,8) if bin(a^b).count("1")==1]
R,I=eqres(cube,0,7);check('网络方程','W10立方体',R,5/6)
V,_=network(cube,{0:1,7:0})
for i in [1,2,4]:check('网络方程','立方体首层等势',V[i],.6)
phi=(1+math.sqrt(5))/2;X=2
for N in range(1,9):
    check('代数递推',f'梯形网络{N}级误差',1+X/(1+X)-phi,(X-phi)/((1+X)*(1+phi)));X=1+X/(1+X)
check('数字代入','W12检流计微安',.002/(1010+20*.601*.399)*1e6,1.97084,tol=3e-6)
power=[]
for ns in [1,2,3,4,6,12]:
    rr=ns*ns*.2/12;cur=ns*1.5/(.6+rr);power.append((cur*cur*.6,ns))
check('整数比较','T2最优串数',max(power)[1],6)
check('仪表负载','T3电压表',12*(20*100/(20+100))/(10+20*100/(20+100)),7.5)
check('仪表负载','T4外接电流表',1/(1/100+1/1000),1000/11)
check('仪表负载','T4内接电流表',100+1,101)
check('数字代入','T5负载功率',10*(69/53)**2,16.9491,tol=1e-6)
check('积分','T7随温度系数',integral(lambda T:.1e-6*T,300,600),.0135,scale=.0135)
for t in [.01,.2,1,3,10]:
    inputw=integral(lambda z:1-np.exp(-z),0,t);diss=integral(lambda z:(1-np.exp(-z))**2,0,t)
    check('积分','T8漂移动能收支',inputw-diss,.5*(1-math.exp(-t))**2)
check('数字代入','T9双载流子电导',e*(1e22*.14+2e22*.05),384.528)
for a,b in [(1,2),(.3,4),(2,2.1)]:
    check('积分','T10径向电阻',integral(lambda z:1/z,a,b),math.log(b/a))
for delta in [-.1,-.001,.001,.1]:
    edges=[('A','C',1),('C','B',1),('A','D',1),('D','B',1+delta),('C','D',.5)]
    V,_=network(edges,{'A':1,'B':0});expect=-delta/(2*(2+delta)*(.5+.5+(1+delta)/(2+delta)))
    check('网络方程','T11失衡检测',(V['C']-V['D'])/.5,expect)
check('叠加','T12共同电压',(12/2+6/3)/(1/2+1/3+1/6),6+2)
check('功率','T12交叉功率',8**2/6-(6**2+2**2)/6,4)
star=[('A','O',2),('B','O',3),('C','O',6)];tri=[('A','B',6),('B','C',18),('C','A',12)]
for A,B in [('A','B'),('B','C'),('C','A')]:check('网络等效','T13星三角',eqres(star,A,B)[0],eqres(tri,A,B)[0])
grid=[]
for i in range(3):
    for k in range(3):
        if i<2:grid.append((i*3+k,(i+1)*3+k,1))
        if k<2:grid.append((i*3+k,i*3+k+1,1))
check('网络方程','T14九节点',eqres(grid,0,8)[0],1.5)
for k in [0,2,5,8,11]:
    R,I=eqres(grid,0,8);h=1e-4;up=list(grid);down=list(grid);a,b,r=grid[k];up[k]=(a,b,r+h);down[k]=(a,b,r-h)
    check('有限差分','T15瑞利灵敏度',(eqres(up,0,8)[0]-eqres(down,0,8)[0])/(2*h),I[k]**2)
for r,g in [(1,.2),(2,.6),(3,1/3)]:
    check('稳定性','T16回复系数',1/r-g,(1-r*g)/r)
base=[('A','C',2),('C','B',4),('A','D',3),('D','B',3),('A','B',1)]
check('端口测试','T17真实电源关闭',eqres(base,'C','D')[0],137/48)
for rg in [.4,1,137/48,5,20]:
    edges=base+[('C','D',rg)];V,I=network(edges,{'B':0},{'A':12});ig=(V['C']-V['D'])/rg;Is=12-V['A']
    check('网络方程','T17端口电流',ig,1.5/(rg+137/48))
    check('能量收支','T17电源真实功率',12*Is,Is*Is+sum(I[k]**2*edges[k][2] for k in [0,1,2,3,5]))
S=.02;r=2;K=.2;Tc=300
def thermal(m,Q):
    A=K+S*S*Tc/(r*(1+m));B=S*S*(1+2*m)/(2*r*(1+m)**2);D=2*Q/(A+math.sqrt(A*A+4*B*Q));I=S*D/(r*(1+m));return D,I,I*I*m*r
for Q in [10,20,50]:
    for m in [.2,1,1.336970172,3,10]:
        D,I,P=thermal(m,Q);Qh=K*D+S*(Tc+D)*I-.5*r*I*I;Qc=K*D+S*Tc*I+.5*r*I*I
        check('热平衡','T18给定供热',Qh,Q);check('热平衡','T18冷热端能量',Qh-Qc,P)
    lo,hi=.01,10
    for _ in range(80):
        x1=lo+(hi-lo)*.38196601125;x2=lo+(hi-lo)*.61803398875
        if thermal(x1,Q)[2]>thermal(x2,Q)[2]:hi=x2
        else:lo=x1
    opt=(lo+hi)/2;D,I,P=thermal(opt,Q)
    check('独立优化','T18黄金分割与驻值',opt,1+S*S/(r*K)*(Tc+D/(1+opt)),tol=1e-7)
check('数字代入','T18最大输出',thermal(1.3369701720267786,20)[2],.3654724601683)
groups={}
for q in records:
    g=groups.setdefault(q['group'],dict(count=0,max_error=0));g['count']+=1;g['max_error']=max(g['max_error'],q['normalized_error'])
report=dict(count=len(records),count_description='独立比较的参数实例数；不是独立公式数量',all_passed=all(q['passed'] for q in records),groups=groups,records=records)
out=Path(__file__).resolve().parent/'检查/第二十讲_公式核验.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False))
assert report['all_passed'],[q for q in records if not q['passed']]
