"""独立数值交叉核验：归一化 k=1，故 epsilon=1/(4 pi)。
不替代纸面证明；保留积分、点对求和、边界和虚功的独立计算路径。
"""
from pathlib import Path
import math,json,itertools,random
ROOT=Path(__file__).resolve().parent
checks=[]
eps=1/(4*math.pi)
def check(name,actual,expected,tol=2e-7):
    err=abs(actual-expected)/max(1,abs(expected))
    checks.append({'name':name,'actual':actual,'expected':expected,'normalized_error':err,'tolerance':tol})
    assert err<tol,(name,actual,expected,err)
def derivative(f,x):
    h=1e-5*max(1,abs(x))
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
def integral(f,a,b,n=4000):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+4*sum(f(a+h*i) for i in range(1,n,2))+2*sum(f(a+h*i) for i in range(2,n,2)))
def norm(v):return math.sqrt(sum(x*x for x in v))
def potential(pos,sources):
    return sum(q/norm(tuple(x-y for x,y in zip(pos,p))) for q,p in sources)
def force(pos,q,sources):
    out=[0.,0.,0.]
    for qi,p in sources:
        d=[x-y for x,y in zip(pos,p)];r=norm(d)
        for i in range(3):out[i]+=q*qi*d[i]/r**3
    return out
def bisect(f,a,b):
    assert f(a)*f(b)<0
    for _ in range(90):
        m=(a+b)/2
        if f(a)*f(m)<=0:b=m
        else:a=m
    return (a+b)/2

# 逐对三维求和，并核对各类连线数量。
b=1.7;e=.8
verts=list(itertools.product([-b/2,b/2],repeat=3))
counts={1:0,2:0,3:0}
for u,v in itertools.combinations(verts,2):
    nd=sum(x!=y for x,y in zip(u,v));counts[nd]+=1
assert counts=={1:12,2:12,3:4}
for lam in [0,1,2,3]:
    src=[(-e,v) for v in verts]+[(lam*e,(0,0,0))]
    pair=sum(q1*q2/norm(tuple(x-y for x,y in zip(p1,p2))) for (q1,p1),(q2,p2) in itertools.combinations(src,2))
    expected=e*e/b*(12+6*math.sqrt(2)+4/math.sqrt(3)-16*lam/math.sqrt(3))
    check(f'cube lambda={lam}',pair,expected)
    for i,F in enumerate(force((0,0,0),lam*e,src[:-1])):check(f'cube center component {lam},{i}',F,0)
assert 12+6*math.sqrt(2)-28/math.sqrt(3)>0

# 内外场积分 vs 装配、解析公式。
Q=1.9;R=2.3
for n in [0,1,2,4,10]:
    q=lambda r:Q*(r/R)**(n+3)
    interior=.5*integral(lambda r:0 if r==0 else q(r)**2/r**2,0,R)
    field=interior+Q*Q/(2*R)
    assembly=integral(lambda r:0 if r==0 else q(r)/r*(n+3)*Q*r**(n+2)/R**(n+3),0,R)
    expected=Q*Q/R*(n+3)/(2*n+5)
    check(f'power density n={n} field',field,expected)
    check(f'power density n={n} assembly',assembly,expected)
for a,b in [(.4,2.1),(1.1,2.5),(1.7,1.9)]:
    D=b**3-a**3
    enclosed=lambda r:Q*(r**3-a**3)/D
    field=.5*integral(lambda r:enclosed(r)**2/r**2,a,b)+Q*Q/(2*b)
    exact=3*Q*Q*(2*b**5-5*a**3*b*b+3*a**5)/(10*D*D)
    check(f'hollow shell {a},{b}',field,exact)
for Q1,Q2 in [(1.2,2.1),(1.2,-1.2),(1.2,-.5)]:
    a,b=1.1,2.6
    field=.5*Q1**2*(1/a-1/b)+.5*(Q1+Q2)**2/b
    exact=Q1**2/(2*a)+Q2**2/(2*b)+Q1*Q2/b
    check(f'two shells {Q1},{Q2}',field,exact)

# 有限偶极的直接两端力 vs 整体位置能梯度。
Qs,q,ell=1.7,.6,.3
for r in [1.,2.,4.]:
    W=lambda x:Qs*q*(1/(x+ell/2)-1/(x-ell/2))
    direct=Qs*q/(r+ell/2)**2-Qs*q/(r-ell/2)**2
    check(f'finite dipole force r={r}',-derivative(W,r),direct)
    check(f'finite dipole energy r={r}',W(r),-Qs*q*ell/(r*r-ell*ell/4))
alpha,E0,g,kappa=.4,1.3,.2,1.1
zstar=alpha*g*E0/(kappa-alpha*g*g)
check('induced particle mechanical equilibrium',alpha*g*(E0+g*zstar)-kappa*zstar,0)
for z in [-.2,0,.7]:
    p=alpha*(E0+g*z)
    internal=p*p/(2*alpha)-p*(E0+g*z)
    check(f'induced eliminate p z={z}',internal,-alpha*(E0+g*z)**2/2)
    check(f'induced force z={z}',-derivative(lambda x:-alpha*(E0+g*x)**2/2,z),alpha*g*(E0+g*z))
for r in [1.2,2.7]:
    W=lambda x:-alpha*Qs*Qs/(2*x**4)
    check(f'induced point source r={r}',-derivative(W,r),-2*alpha*Qs*Qs/r**5)

# 恒压与恒电荷，用瞬时电压、电量交换独立构造机械微元。
S,V,Q=2.4,1.6,1.1
for d in [.8,1.7,2.6]:
    C=eps*S/d
    FQ=-derivative(lambda x:Q*Q*x/(2*eps*S),d)
    FV=-derivative(lambda x:-eps*S*V*V/(2*x),d)
    check(f'plate Q pressure d={d}',FQ,-eps*S*(Q/(eps*S))**2/2)
    check(f'plate V pressure d={d}',FV,-eps*S*(V/d)**2/2)
d0,d1=2.,.9
source=V*V*eps*S*(1/d1-1/d0)
mechanical=integral(lambda d:-eps*S*V*V/(2*d*d),d0,d1)
deltafield=.5*eps*S*V*V*(1/d1-1/d0)
check('constant V plate power balance',source,mechanical+deltafield)
sigma=.7;R0,R1=1.,2.1
source=integral(lambda r:(sigma*r/eps)*(8*math.pi*sigma*r),R0,R1)
field=2*math.pi*sigma*sigma/eps*(R1**3-R0**3)
mechanical=integral(lambda r:(sigma*sigma/(2*eps))*4*math.pi*r*r,R0,R1)
check('constant sigma source balance',source,field+mechanical)
check('constant sigma mechanical work',mechanical,2*math.pi*sigma*sigma/(3*eps)*(R1**3-R0**3))
for C1,C2,V1,V2 in [(1,2,2,1),(1,2,2,-1),(.4,3,1.2,-.3),(2,1,2,2)]:
    vf=(C1*V1+C2*V2)/(C1+C2)
    wi=(C1*V1*V1+C2*V2*V2)/2
    wf=(C1+C2)*vf*vf/2
    exact=C1*C2*(V1-V2)**2/(2*(C1+C2))
    check(f'sharing {C1},{C2},{V1},{V2}',wi-wf,exact)

# 接地平面：边界、密度法向导数、面积积分、力和能量。
q=1.3
for h in [.7,1.4,2.8]:
    src=[(q,(0,0,h)),(-q,(0,0,-h))]
    for rho in [0,.3,2.1]:
        check(f'plane boundary h={h},rho={rho}',potential((rho,0,0),src),0)
        dz=derivative(lambda z:potential((rho,0,z),src),0)
        sigma=-eps*dz
        exact=-q*h/(2*math.pi*(rho*rho+h*h)**1.5)
        check(f'plane density h={h},rho={rho}',sigma,exact)
    direct=force((0,0,h),q,src[1:])[2]
    check(f'plane image force h={h}',direct,-q*q/(4*h*h))
    check(f'plane energy derivative h={h}',-derivative(lambda z:-q*q/(4*z),h),direct)
    # rho=h tan(t); 变换后感应电量的面积积分为 -q sin(t)。
    check(f'plane total induced h={h}',integral(lambda t:-q*math.sin(t),0,math.pi/2),-q)

# 接地/中性球边界距离比和能量导数 vs 真实点源处镜像库仑力。
a=1.1
for s in [1.4,2.,4.5]:
    qp=-q*a/s;bp=a*a/s
    grounded=[(q,(s,0,0)),(qp,(bp,0,0))]
    neutral=grounded+[(q*a/s,(0,0,0))]
    for theta in [.1,.7,1.4,2.7]:
        P=(a*math.cos(theta),a*math.sin(theta),0)
        check(f'sphere boundary ground s={s},t={theta}',potential(P,grounded),0)
        check(f'sphere boundary neutral s={s},t={theta}',potential(P,neutral),q/s)
        RR=norm(tuple(P[i]-grounded[0][1][i] for i in range(3)))
        RI=norm(tuple(P[i]-grounded[1][1][i] for i in range(3)))
        check(f'sphere ratio s={s},t={theta}',RI/RR,a/s)
    for mode,src,W in [('ground',grounded,lambda x:-q*q*a/(2*(x*x-a*a))),
                       ('neutral',neutral,lambda x:-q*q*a**3/(2*x*x*(x*x-a*a)))]:
        direct=force((s,0,0),q,src[1:])[0]
        check(f'sphere force derivative {mode},s={s}',-derivative(W,s),direct)
        check(f'sphere half induced {mode},s={s}',W(s),q*potential((s,0,0),src[1:])/2)

# 双平面：边界上四源直接求势，二维能量导数对三镜像力。
for a,b in [(.8,1.4),(1.3,1.3),(2.6,.9)]:
    src=[(q,(a,b,0)),(-q,(-a,b,0)),(-q,(a,-b,0)),(q,(-a,-b,0))]
    for P in [(0,.2,.5),(0,2.7,.3),(.4,0,.6),(2.4,0,0)]:
        check(f'corner boundary {a},{b},{P}',potential(P,src),0)
    W=lambda x,y:q*q/4*(-1/x-1/y+1/math.hypot(x,y))
    F=force((a,b,0),q,src[1:])
    check(f'corner Fx {a},{b}',-derivative(lambda x:W(x,b),a),F[0])
    check(f'corner Fy {a},{b}',-derivative(lambda y:W(a,y),b),F[1])
    check(f'corner energy half {a},{b}',W(a,b),q*potential((a,b,0),src[1:])/2)
u=1.4
A0=q*q*(2-1/math.sqrt(2))/4
Fu=-A0/u**2
check('diagonal virtual work metric',Fu,-A0/(math.sqrt(2)*u*u)*math.sqrt(2))

# 两点源与平面：改变第二点源 x 的直接力 vs 完整能量梯度。
for q1,q2 in [(1.2,2.1),(1.2,-.8)]:
    h,ell=1.3,1.7;R=math.hypot(ell,2*h)
    W=lambda l,H:q1*q2/l-(q1*q1+q2*q2)/(4*H)-q1*q2/math.hypot(l,2*H)
    ind1=-q1/(2*h)-q2/R;ind2=-q2/(2*h)-q1/R
    check(f'two sources energy {q2}',W(ell,h),q1*q2/ell+(q1*ind1+q2*ind2)/2)
    F2=force((ell,0,h),q2,[(q1,(0,0,h)),(-q1,(0,0,-h)),(-q2,(ell,0,-h))])
    F1=force((0,0,h),q1,[(q2,(ell,0,h)),(-q1,(0,0,-h)),(-q2,(ell,0,-h))])
    check(f'two sources Fx {q2}',-derivative(lambda l:W(l,h),ell),F2[0])
    check(f'two sources common h {q2}',-derivative(lambda H:W(ell,H),h),F1[2]+F2[2])

# 同轴压力核对。
for a,b in [(.7,2),(1.1,2.6)]:
    L,Q=3.1,1.6
    W=lambda x:Q*Q/(4*math.pi*eps*L)*math.log(b/x)
    p=(Q/(2*math.pi*a*L))**2/(2*eps)
    check(f'coax pressure a={a}',-derivative(W,a),p*2*math.pi*a*L)

# 恒压吸合：直接解力平衡根，再按有限扰动判断曲率。
d0,kappa,S,m=2.1,1.4,2.3,.9
Vc=math.sqrt(8*kappa*d0**3/(27*eps*S))
f=lambda x:x*(d0-x)**2
check('pullin f maximum derivative',derivative(f,d0/3),0)
check('pullin V critical balance',kappa*d0/3,eps*S*Vc*Vc/(2*(d0-d0/3)**2))
for ratio in [.2,.6,.9,.99]:
    V=ratio*Vc
    force_total=lambda x:eps*S*V*V/(2*(d0-x)**2)-kappa*x
    low=bisect(force_total,0,d0/3);high=bisect(force_total,d0/3,d0*(1-1e-7))
    for label,x in [('stable',low),('unstable',high)]:
        stiffness=-derivative(force_total,x)
        expected=kappa*(d0-3*x)/(d0-x)
        check(f'pullin stiffness {ratio},{label}',stiffness,expected)
        assert (stiffness>0)==(label=='stable')
    check(f'pullin frequency squared {ratio}',-derivative(force_total,low)/m,
          kappa/m*(d0-3*low)/(d0-low))
result={'units':'k=1, epsilon=1/(4pi)','count':len(checks),'all_passed':True,
        'maximum_normalized_error':max(c['normalized_error'] for c in checks),
        'limits':'数值交叉核验覆盖上述有限模型，不替代唯一性、收敛性及近似条件的解析论证。',
        'checks':checks}
(ROOT/'检查/第六讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
