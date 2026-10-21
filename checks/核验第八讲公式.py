"""独立求积、叉乘、边界与极限核验；所有长度、电流以归一单位计算。"""
from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parent
tests=[]
X,W=leggauss(120)
def integral(f,a,b):
    t=(a+b)/2+(b-a)/2*X
    v=np.asarray(f(t))
    return (b-a)/2*np.tensordot(W,v,axes=(0,0))
def check(name,value,expected,tol=2e-9):
    value=np.asarray(value);expected=np.asarray(expected)
    err=float(np.max(np.abs(value-expected))/(max(1,float(np.max(np.abs(expected))))))
    assert err<tol,(name,value,expected,err,tol)
    tests.append(dict(name=name,normalized_error=err,tolerance=tol))
def line_B(A,D,P,I=1):
    A,D,P=map(np.asarray,(A,D,P)); dl=D-A
    def f(t):
        R=P-(A+t[:,None]*dl)
        return I*np.cross(dl,R)/np.linalg.norm(R,axis=1)[:,None]**3
    return integral(f,0,1)
def arc_B(radius,start,end,P=(0,0,0),I=1):
    P=np.asarray(P)
    def f(t):
        pts=np.column_stack((radius*np.cos(t),radius*np.sin(t),np.zeros_like(t)))
        dl=np.column_stack((-radius*np.sin(t),radius*np.cos(t),np.zeros_like(t)))
        R=P-pts
        return I*np.cross(dl,R)/np.linalg.norm(R,axis=1)[:,None]**3
    return integral(f,start,end)
for beta in [.05,.4,1.3,4]:
    U,L,S,gam,eps=2.4,1.7,.6,3.2,1.5
    I=gam*S*beta*U/(L*math.log1p(beta))
    E=lambda z:I/(gam*S*(1+beta*z/L))
    rho=lambda z:-eps*I*beta/(gam*S*L*(1+beta*z/L)**2)
    check('W1 电压积分 β='+str(beta),integral(E,0,L),U)
    check('W1 体电荷积分 β='+str(beta),S*integral(rho,0,L),
          -eps*S*beta**2*U/(L*math.log1p(beta)*(1+beta)))
    check('W1 边界高斯关系 β='+str(beta),S*integral(rho,0,L),eps*S*(E(L)-E(0)))
for a,b in [(.4,1.3),(.9,1.8),(1.2,4.2)]:
    A0,E0,Vc=2.2,.4,-.7
    phi=lambda r,z:Vc+(A0-E0*z)*math.log(b/r)/math.log(b/a)
    for z in [-.3,.7,2.5]:
        check('同轴内边界',phi(a,z),Vc+A0-E0*z)
        check('同轴外边界',phi(b,z),Vc)
        r=(a+b)/2;h=2e-4
        dr=(phi(r+h,z)-phi(r-h,z))/(2*h)
        drr=(phi(r+h,z)-2*phi(r,z)+phi(r-h,z))/h**2
        dzz=(phi(r,z+h)-2*phi(r,z)+phi(r,z-h))/h**2
        check('同轴拉普拉斯差分',drr+dr/r+dzz,0,tol=2e-6)
    check('同轴表面电量相反',a*A0/(a*math.log(b/a)),b*A0/(b*math.log(b/a)))
for aa,bb,h in [(.5,1.2,.8),(1,1,1),(2,.3,.4),(4,8,1.4)]:
    B=line_B((-aa,0,0),(bb,0,0),(0,h,0))
    ans=(aa/math.sqrt(aa*aa+h*h)+bb/math.sqrt(bb*bb+h*h))/h
    check('W5 有限导线的三维求积',B,[0,0,ans])
for a,L,h in [(.4,1.3,.6),(1,3,.9),(.2,1.4,1.8)]:
    ans=((a+L)/math.sqrt((a+L)**2+h*h)-a/math.sqrt(a*a+h*h))/h
    check('T6 外侧垂足求积',line_B((0,0,0),(L,0,0),(-a,h,0)),[0,0,ans])
for a,b,Phi in [(.5,1.7,1.4),(1,2,2.6),(.9,1.1,4.8)]:
    inside=arc_B(a,0,Phi);outside=arc_B(b,Phi,0)
    rA=np.array([a,0,0]);rB=np.array([b,0,0])
    rC=np.array([b*math.cos(Phi),b*math.sin(Phi),0])
    rD=np.array([a*math.cos(Phi),a*math.sin(Phi),0])
    connectors=line_B(rB,rA,(0,0,0))+line_B(rD,rC,(0,0,0))
    check('W6 两圆弧与径向段求积',inside+outside+connectors,[0,0,Phi*(1/a-1/b)])
    # 实际有向回路：内弧0→Phi、上径向向外、外弧Phi→0、下径向向内。
    def moment_arc(r,t0,t1):
        return .5*integral(lambda t:np.full_like(t,r*r),t0,t1)
    check('W6 磁矩方向与扇环面积',moment_arc(a,0,Phi)+moment_arc(b,Phi,0),-.5*Phi*(b*b-a*a))
for a,b in [(1,1),(.6,1.7),(2,.5)]:
    pts=[(-a,-b,0),(a,-b,0),(a,b,0),(-a,b,0)]
    B=sum(line_B(pts[i],pts[(i+1)%4],(0,0,0)) for i in range(4))
    check('W7 矩形四边求积',B,[0,0,4*math.sqrt(a*a+b*b)/(a*b)])
for z in [0,.1,.4,1.2,3]:
    check('圆线圈轴上场三维求积',arc_B(1.3,0,2*math.pi,(0,0,z)),[0,0,2*math.pi*1.3**2/(1.3**2+z*z)**1.5])
for a,b in [(.7,1.5),(1,3)]:
    Ib=b/a
    field=lambda z:2*math.pi*(a*a/(a*a+z*z)**1.5-Ib*b*b/(b*b+z*z)**1.5)
    check('W8 中心反绕线圈零场',field(0),0)
    z=1e-3*a
    leading=-3*math.pi/a*(1/a**2-1/b**2)*z*z
    check('W8 中心二阶首项',field(z)/z**2,leading/z**2,tol=6e-6)
for d,w,L in [(.6,.5,1.4),(1,1,2.3),(1.8,.3,.7)]:
    # 直接把无穷直导线场Bz=2/y作用在回路各边上。
    pts=[np.array(p) for p in [(0,d,0),(L,d,0),(L,d+w,0),(0,d+w,0)]]
    F=np.zeros(3)
    for i in range(4):
        A,D=pts[i],pts[(i+1)%4];dl=D-A
        F+=integral(lambda t:np.cross(np.broadcast_to(dl,(len(t),3)),
                    np.column_stack((np.zeros_like(t),np.zeros_like(t),2/(A[1]+t*dl[1])))),0,1)
    check('W10 四边受力积分',F,[0,-2*L*(1/d-1/(d+w)),0])
for lam in [.4,1.6,3.2]:
    A,D,P=np.array([-1,0,0]),np.array([2,0,0]),np.array([.3,.8,.4])
    B=line_B(A,D,P)
    B2=line_B(lam*A,lam*D,lam*P)
    check('W11 几何缩放场',B2,B/lam)
    test=np.array([.2,-.3,.6])
    check('W11 几何缩放力',np.cross(lam*test,B2),np.cross(test,B))
for a,r in [(.2,1),(1,2),(.7,.8)]:
    P=np.array([0,r,0]);R1=P-np.array([-a,0,0]);R2=P-np.array([a,0,0])
    # kappa=1，无限直线场=2I z×R/R²。
    B1=2*np.cross([0,0,1],R1)/(a*a+r*r)
    B2=2*np.cross([0,0,-1],R2)/(a*a+r*r)
    check('W12 反向双线分量相加',B1+B2,[0,4*a/(a*a+r*r),0])
for alpha,theta in [(1.1,.3),(.8,.2),(math.pi/2,.5)]:
    BH=2.4;Bw=BH*math.sin(theta)/math.sin(alpha-theta)
    check('T4 非正交磁针矢量反解',math.atan2(Bw*math.sin(alpha),BH+Bw*math.cos(alpha)),theta)
check('T5 数值叉乘',2*np.cross([.03,.04,0],[0,0,.20]),[.016,-.012,0])
for a in [.4,1,2.3]:
    # t/(1-t)把半无限线直接映射到有限求积区间。
    def semi_x(t):
        s=t/(1-t);den=((a+s)**2+a*a)**1.5
        return -a/den/(1-t)**2
    check('T7 折角半无限线求积',2*integral(semi_x,0,1),-2/a*(1-1/math.sqrt(2)))
for a in [.6,1,2.5]:
    arc=arc_B(a,math.pi/2,2*math.pi)
    chord=line_B((a,0,0),(0,a,0),(0,0,0))
    check('T8 弧加弦完整积分',arc+chord,[0,0,(1.5*math.pi+2)/a])
for n in [3,4,6,9,16]:
    R=1.3;pts=[(R*math.cos(2*math.pi*i/n),R*math.sin(2*math.pi*i/n),0) for i in range(n)]
    B=sum(line_B(pts[i],pts[(i+1)%n],(0,0,0)) for i in range(n))
    check('T9 多边形逐边求积 n='+str(n),B,[0,0,2*n/R*math.tan(math.pi/n)])
check('T10 垂直场合成',np.linalg.norm([2*math.pi,2*math.pi,0]),2*math.pi*math.sqrt(2))
for a,b in [(.6,1.4),(1,2.8),(.9,1.1)]:
    check('T11 环带求积',integral(lambda r:2*math.pi/((b-a)*r),a,b),2*math.pi/(b-a)*math.log(b/a))
for R,sigma,omega in [(.8,1.3,2),(1.7,.4,.8)]:
    check('T12 圆盘磁场求积',integral(lambda r:np.full_like(r,2*math.pi*sigma*omega),0,R),2*math.pi*sigma*omega*R)
    check('T12 圆盘磁矩求积',integral(lambda r:math.pi*sigma*omega*r**3,0,R),math.pi*sigma*omega*R**4/4)
check('T13 第一次力',.2*np.cross([1,0,0],[.01,.02,.03]),[0,-.006,.004])
check('T13 第二次力',.2*np.cross([0,1,0],[.01,.02,.03]),[.006,0,-.002])
for P in [(3,0,40),(20,25,35),(30,0,0)]:
    P=np.array(P,float);r=np.linalg.norm(P);m=np.array([0,0,math.pi])
    dip=(3*np.dot(m,P)*P/r**2-m)/r**3
    exact=arc_B(1,0,2*math.pi,P)
    check('T14 远場偶极展开',exact*r**3,dip*r**3,tol=.007)
    # 闭合积分恒等式独立验证
    ident=integral(lambda t:((P[0]*np.cos(t)+P[1]*np.sin(t))[:,None]*
                     np.column_stack((-np.sin(t),np.cos(t),np.zeros_like(t)))),0,2*math.pi)
    check('T14 闭合回路磁矩恒等式',ident,np.cross(m,P))
def loop_nodes(center,axis1,axis2,R=1,N=100):
    t=(np.arange(N)+.5)*2*math.pi/N
    center=np.array(center);axis1=np.array(axis1);axis2=np.array(axis2)
    pts=center+R*(np.cos(t)[:,None]*axis1+np.sin(t)[:,None]*axis2)
    dl=R*2*math.pi/N*(-np.sin(t)[:,None]*axis1+np.cos(t)[:,None]*axis2)
    return pts,dl
for sep in [2.8,4.3]:
    P1,l1=loop_nodes((0,0,0),(1,0,0),(0,1,0),.7)
    P2,l2=loop_nodes((sep,.3,.8),(0,1,0),(0,0,1),.6)
    R=P2[:,None,:]-P1[None,:,:];den=np.linalg.norm(R,axis=2)**3
    B21=np.sum(np.cross(l1[None,:,:],R)/den[:,:,None],axis=1)
    B12=np.sum(np.cross(l2[:,None,:],-R)/den[:,:,None],axis=0)
    F2=np.sum(np.cross(l2,B21),axis=0);F1=np.sum(np.cross(l1,B12),axis=0)
    check('T15 完整不共面双回路总力互易',F1+F2,[0,0,0])
for B in [[.7,.2,1.3],[2,-.4,.8]]:
    pts,dl=loop_nodes((0,0,0),(1,0,0),(0,1,0),1.3)
    force=np.cross(dl,B)
    check('T16 均匀场闭合净力',np.sum(force,axis=0),[0,0,0])
    check('T16 力矩积分',np.sum(np.cross(pts,force),axis=0),np.cross([0,0,math.pi*1.3**2],B))
f=lambda s:(1+s*s)**-1.5
for d in [.3,.5,.8]:
    h=1e-3
    calc=(f(d+h)-2*f(d)+f(d-h))/h**2
    exact=3*(4*d*d-1)/(1+d*d)**3.5
    check('T17 轴场曲率差分',calc,exact,tol=5e-6)
for z in [.02,.04,.07]:
    ratio=(f(z-.5)+f(z+.5))/(2*f(.5))
    c4=(ratio-1)/z**4
    check('T17 四阶系数及高阶误差',c4,-144/125,tol=.006)
g=lambda t:t*(1-t)*(2-t)
tc=1-1/math.sqrt(3);lc=2/(3*math.sqrt(3))
check('T18 临界极值',g(tc),lc)
for t in [.1,.3,.65,.85]:
    H=lambda t:(1-t)*(2-t);Lam=g(t);h=1e-5
    F=lambda t:Lam/H(t)
    stiff=1-(F(t+h)-F(t-h))/(2*h)
    exact=(2-6*t+3*t*t)/H(t)
    check('T18 净力导数与稳定刚度',stiff,exact,tol=3e-7)
    check('T18 无量纲平衡方程',F(t),t)
report={'lecture':'第八讲','passed':len(tests),'max_normalized_error':max(t['normalized_error'] for t in tests),'checks':tests}
(ROOT/'检查/第八讲_公式核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))
