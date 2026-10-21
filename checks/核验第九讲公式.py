"""以直接源积分、差分、穿面电流和矢量变换核验第九讲；归一化 mu0=1。"""
from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parent
X,W=leggauss(160)
tests=[]
def quad(f,a,b):
    t=(a+b)/2+(b-a)/2*X
    return (b-a)/2*np.tensordot(W,np.asarray(f(t)),axes=(0,0))
def check(name,v,e,tol=2e-9,kind='积分或恒等式'):
    v,e=map(np.asarray,(v,e))
    err=float(np.max(np.abs(v-e))/max(1,float(np.max(np.abs(e)))))
    assert err<tol,(name,v,e,err,tol)
    tests.append(dict(name=name,normalized_error=err,tolerance=tol,kind=kind))
def loop(R,z0,P,I=1):
    P=np.asarray(P)
    def f(t):
        src=np.column_stack((R*np.cos(t),R*np.sin(t),np.full_like(t,z0)))
        dl=np.column_stack((-R*np.sin(t),R*np.cos(t),np.zeros_like(t)))
        rv=P-src
        return I*np.cross(dl,rv)/(4*math.pi*np.linalg.norm(rv,axis=1)[:,None]**3)
    return quad(f,0,2*math.pi)
def line(z1,z2,r,I=1):
    return quad(lambda z:I*r/(4*math.pi*(r*r+z*z)**1.5),z1,z2)
def sol(R,h,z,n=1,I=1):
    return n*I/2*((z+h)/math.sqrt(R*R+(z+h)**2)-(z-h)/math.sqrt(R*R+(z-h)**2))
for a,b,r in [(.4,2,.7),(1,4,1.3),(-2,3,1),(-3,-.5,.6)]:
    check('直线段有向端点',line(a,b,r),
          (b/math.sqrt(r*r+b*b)-a/math.sqrt(r*r+a*a))/(4*math.pi*r))
for R in [.6,1,1.7]:
    for z in [-2,-.4,0,.8,3]:
        check('圆线圈三维毕奥积分',loop(R,0,(0,0,z)),[0,0,R*R/(2*(R*R+z*z)**1.5)])
for R,h in [(.6,.4),(1,2),(1,8)]:
    for z in [-h-.7,0,.6*h,h,h+.9]:
        value=quad(lambda zs:R*R/(2*(R*R+(z-zs)**2)**1.5),-h,h)
        check('有限螺线管逐匝积分',value,sol(R,h,z))
    check('中心和端面场比',sol(R,h,h)/sol(R,h,0),math.sqrt(R*R+h*h)/math.sqrt(R*R+4*h*h))
check('1%中心精度边界',sol(1,.99/math.sqrt(1-.99**2),0),.99)
for h,np_,nm in [(.5,2,1),(2,3,.5),(5,1,4)]:
    R=1.3
    B=lambda z:quad(lambda zs:np_*R*R/(2*(R*R+(z-zs)**2)**1.5),0,h)+quad(lambda zs:nm*R*R/(2*(R*R+(z-zs)**2)**1.5),-h,0)
    check('非均匀匝密度中心',B(0),(np_+nm)/2*h/math.sqrt(R*R+h*h))
    eps=1e-4
    check('非均匀匝密度梯度差分',(B(eps)-B(-eps))/(2*eps),
          (np_-nm)/2*(1/R-R*R/(R*R+h*h)**1.5),tol=3e-8,kind='差分')
for R,d in [(1,.3),(1,.5),(1,.8),(1.7,.9)]:
    B=lambda z:loop(R,d,(0,0,z))[2]+loop(R,-d,(0,0,z))[2]
    h=2e-4
    curv=(B(h)-2*B(0)+B(-h))/h**2
    expected=3*R*R*(4*d*d-R*R)/(R*R+d*d)**3.5
    check('同向线圈曲率三维源积分',curv,expected,tol=8e-8,kind='差分')
    BA=lambda z:loop(R,d,(0,0,z))[2]-loop(R,-d,(0,0,z))[2]
    check('反向线圈梯度三维源积分',(BA(h)-BA(-h))/(2*h),
          3*R*R*d/(R*R+d*d)**2.5,tol=8e-8,kind='差分')
    g=3*R*R*d/(R*R+d*d)**2.5
    rho=1e-4
    br=loop(R,d,(rho,0,0))[0]-loop(R,-d,(rho,0,0))[0]
    check('反向线圈横向梯度',br/rho,-g/2,tol=3e-8,kind='差分')
g=lambda d:3*d/(1+d*d)**2.5
for d in [.1,.3,.49,.51,.8,2]:
    assert g(d)<g(.5)
check('反向线圈最大梯度',g(.5),1.5*(4/5)**2.5)
# 渐近展开均指定有限位置的误差容差，单独列类。
BH=(4/5)**1.5
c=144/125
for rho,z in [(0,.03),(.03,0),(.025,.02),(.04,-.02)]:
    exact=loop(1,.5,(rho,0,z))+loop(1,-.5,(rho,0,z))
    bz=BH*(1-c*(z**4-3*rho*rho*z*z+3*rho**4/8))
    br=BH*c*(2*rho*z**3-1.5*rho**3*z)
    check('亥姆霍兹三维四阶展开',exact,[br,0,bz],tol=2e-8,kind='渐近截断')
for z in [-1.7,-.4,.3,2.2]:
    R=1.2
    omega=quad(lambda s:2*math.pi*z*s/(s*s+z*z)**1.5,0,R)
    check('圆盘立体角源积分',omega,2*math.pi*(np.sign(z)-z/math.sqrt(R*R+z*z)))
    h=1e-4
    om=lambda z:quad(lambda s:2*math.pi*z*s/(s*s+z*z)**1.5,0,R)
    check('立体角梯度等于圆回路场',-(om(z+h)-om(z-h))/(8*math.pi*h),
          loop(R,0,(0,0,z))[2],tol=3e-8,kind='差分')
for mode in ['r2','center']:
    a=1.4;J0=2.3
    J=(lambda s:J0*s*s/a**2) if mode=='r2' else (lambda s:J0*(1-s*s/a**2))
    I=quad(lambda s:2*math.pi*J(s)*s,0,a)
    check('径向分布总电流 '+mode,I,math.pi*J0*a*a/2)
    for r in [.1*a,.5*a,.9*a,a,1.3*a]:
        B=quad(lambda s:J(s)*s,0,min(r,a))/r
        inside=J0*r**3/(4*a*a) if mode=='r2' else J0/2*(r-r**3/(2*a*a))
        ans=inside if r<a else I/(2*math.pi*r)
        check('径向累计电流求场 '+mode,B,ans)
    if mode=='center':
        rs=a*math.sqrt(2/3)
        check('内部峰值',quad(lambda s:J(s)*s,0,rs)/rs,J0*a/3*math.sqrt(2/3))
for a,b,c in [(.5,1,1.5),(1,1.3,2.4)]:
    I=2.4
    for r in [.2*a,a,(a+b)/2,b,(b+c)/2,c,1.2*c]:
        enc=I*min(r*r/(a*a),1)
        if r>b:enc-=I*min((r*r-b*b)/(c*c-b*b),1)
        ans=I*r/(2*math.pi*a*a) if r<a else I/(2*math.pi*r) if r<b else I*(c*c-r*r)/(2*math.pi*r*(c*c-b*b)) if r<c else 0
        check('均匀同轴回流累计',enc/(2*math.pi*r),ans)
    A=I/(2*math.pi*math.log(c/b))
    check('非均匀回流总量',quad(lambda s:2*math.pi*A/s,b,c),I)
    for r in [b,(b+c)/2,c]:
        enc=I-quad(lambda s:2*math.pi*A/s,b,r)
        check('非均匀回流包围电流',enc,I*math.log(c/r)/math.log(c/b))
    r=(b+c)/2;step=1e-5
    rb=lambda r:I/(2*math.pi)*math.log(c/r)/math.log(c/b)
    check('非均匀回流旋度差分',(rb(r+step)-rb(r-step))/(2*step*r),-A/r**2,tol=3e-9,kind='差分')
for D,a in [(1.4,1),(2,1),(3.5,.8)]:
    s2=lambda t:D*D+a*a-2*D*a*np.cos(t)
    bt=lambda t:(a-D*np.cos(t))/(2*math.pi*s2(t))
    check('外部导线完整环流',quad(lambda t:a*bt(t),0,2*math.pi),0)
    check('外部导线近点切向',bt(0),-1/(2*math.pi*(D-a)))
    check('外部导线远点切向',bt(math.pi),1/(2*math.pi*(D+a)))
for a in [.4,1,2]:
    J=1.7
    for z in [-1.3*a,-a,-.4*a,0,.4*a,a,1.3*a]:
        # 无穷薄电流面叠加，积分在场点处分开，避免不连续求积。
        low=min(max(z,-a),a)
        by=-.5*quad(lambda u:np.full_like(u,J),-a,low)+.5*quad(lambda u:np.full_like(u,J),low,a)
        ans=J*a if z<-a else -J*z if abs(z)<=a else -J*a
        check('均匀电流层逐面叠加',by,ans)
        odd=-.5*quad(lambda u:J*u/a,-a,low)+.5*quad(lambda u:J*u/a,low,a)
        ans=J*(a*a-z*z)/(2*a) if abs(z)<a else 0
        check('反向体电流层逐面叠加',odd,ans)
for a,b,h in [(.7,1.3,2),(1,1.1,.5),(1,3,1.7)]:
    NI=2.5
    check('环形截面磁通逐条带积分',quad(lambda r:NI*h/(2*math.pi*r),a,b),NI*h/(2*math.pi)*math.log(b/a))
for beta,gamma,a in [(1,2,.7),(1,0,1),(2,2,1)]:
    Ib=2*math.pi*beta*a*a
    Ks=gamma/a-beta*a
    check('体电流加边界面电流',Ib+2*math.pi*a*Ks,2*math.pi*gamma)
# 均匀无限圆柱的场：场点在柱内时，从场点为中心的射线直接积分。
def disk_inside(radius,P,J=1):
    P=np.asarray(P)
    def f(t):
        er=np.column_stack((np.cos(t),np.sin(t)))
        proj=er@P
        rho=-proj+np.sqrt(radius*radius-np.dot(P,P)+proj*proj)
        rotated=np.column_stack((-er[:,1],er[:,0]))
        return -J/(2*math.pi)*rho[:,None]*rotated
    return quad(f,0,2*math.pi)
def disk_outside(radius,P,J=1):
    P=np.asarray(P)
    def angular(t):
        er=np.column_stack((np.cos(t),np.sin(t)))
        def radial(s):
            rv=P[None,None,:]-s[:,None,None]*er[None,:,:]
            rot=np.stack((-rv[:,:,1],rv[:,:,0]),axis=2)
            return s[:,None,None]*rot/np.sum(rv*rv,axis=2)[:,:,None]
        return J/(2*math.pi)*quad(radial,0,radius)
    return quad(angular,0,2*math.pi)
for P in [[.2,.1],[-.4,.3],[.6,-.1]]:
    check('均匀柱内部直接源积分',disk_inside(1,P),np.array([-P[1],P[0]])/2)
for P in [[1.4,.3],[-2,.8]]:
    expected=np.array([-P[1],P[0]])/(2*np.dot(P,P))
    check('均匀柱外部面积源积分',disk_outside(1,P),expected,tol=2e-8)
R,a,d=2,.65,.7
for sx,sy in [(0,.1),(-.2,.2),(.3,-.15)]:
    P=np.array([d+sx,sy]);s=np.array([sx,sy])
    value=disk_inside(R,P)-disk_inside(a,s)
    check('偏心孔两体源积分',value,[0,d/2],tol=2e-9)
    Iw=.3
    total=value+Iw/(2*math.pi)*np.array([-sy,sx])/np.dot(s,s)
    check('孔内细线分量叠加',total,[-Iw*sy/(2*math.pi*np.dot(s,s)),d/2+Iw*sx/(2*math.pi*np.dot(s,s))])
Iw=.3;sstar=Iw/(math.pi*d)
check('孔内零场点',[0,d/2]+Iw/(2*math.pi)*np.array([0,-sstar])/sstar**2,[0,0])
for s in [.2,.5]:
    check('均匀孔场环流',quad(lambda t:d/2*np.cos(t)*s,0,2*math.pi),0)
    check('孔内细线环流',quad(lambda t:np.full_like(t,Iw/(2*math.pi)),0,2*math.pi),Iw)
# 两孔：第2个负电流圆柱必须按外部源积分处理。
d1=np.array([.7,0]);d2=np.array([-.7,-.9]);a1=.65;a2=.4
for P in [d1+[.1,.15],d1+[-.2,-.1]]:
    B=disk_inside(2,P)-disk_inside(a1,P-d1)-disk_outside(a2,P-d2)
    s2=P-d2
    ans=np.array([-d1[1],d1[0]])/2-a2*a2*np.array([-s2[1],s2[0]])/(2*np.dot(s2,s2))
    check('两孔叠加面积源积分',B,ans,tol=3e-8)
for P in [[.3,.2,.6],[-.7,.1,1.2]]:
    Bp=loop(1,0,P);Pm=np.array(P)*[1,1,-1]
    check('平面电流轴矢量镜像',loop(1,0,Pm),Bp*[-1,-1,1])
    for same in [True,False]:
        B=loop(1,.5,P)+(1 if same else -1)*loop(1,-.5,P)
        Bm=loop(1,.5,Pm)+(1 if same else -1)*loop(1,-.5,Pm)
        check('双线圈同向/反向镜像',Bm,B*np.array([-1,-1,1])*(1 if same else -1))
for theta in [0,.3,.7,1.1]:
    area=math.pi*1.4**2/math.cos(theta)
    check('斜截面投影与穿面电流',area*math.cos(theta),math.pi*1.4**2)
def omega_disk(P,R=1):
    P=np.asarray(P)
    def angular(t):
        er=np.column_stack((np.cos(t),np.sin(t),np.zeros_like(t)))
        def radial(s):
            rv=P[None,None,:]-s[:,None,None]*er[None,:,:]
            return s[:,None]*P[2]/np.linalg.norm(rv,axis=2)**3
        return quad(radial,0,R)
    return quad(angular,0,2*math.pi)
for P in [[.3,.1,.5],[.1,-.6,1.2],[2,.3,-.7]]:
    P=np.asarray(P);step=1e-4
    grad=np.array([(omega_disk(P+step*np.eye(3)[k])-omega_disk(P-step*np.eye(3)[k]))/(2*step) for k in range(3)])
    check('非轴点立体角梯度与毕奥积分',-grad/(4*math.pi),loop(1,0,P),tol=3e-8,kind='差分')
    dr=np.array([.1,-.04,.03])
    def band(t):
        src=np.column_stack((np.cos(t),np.sin(t),np.zeros_like(t)))
        dl=np.column_stack((-np.sin(t),np.cos(t),np.zeros_like(t)))
        area=-np.cross(dr,dl);rv=P-src
        return np.sum(area*rv,axis=1)/np.linalg.norm(rv,axis=1)**3
    check('窄带定向面元三重积',quad(band,0,2*math.pi),np.dot(grad,dr),tol=3e-8,kind='差分')
for h,r,D in [(.6,1,3),(2,.8,2.4)]:
    theta=(X+1)*math.pi
    P=np.column_stack((r*np.cos(theta),r*np.sin(theta),np.zeros_like(theta)))
    dl_circle=np.column_stack((-r*np.sin(theta),r*np.cos(theta),np.zeros_like(theta)))
    vertices=[np.array(q) for q in [(0,0,-h),(0,0,h),(D,0,h),(D,0,-h)]]
    Bs=[]
    for k in range(4):
        A,End=vertices[k],vertices[(k+1)%4];dl=End-A
        t=(X+1)/2;src=A+t[:,None]*dl
        rv=P[:,None,:]-src[None,:,:]
        val=np.cross(dl,rv)/(4*math.pi*np.linalg.norm(rv,axis=2)[:,:,None]**3)
        Bs.append(np.tensordot(val,W/2,axes=(1,0)))
    circulation=lambda B:float(math.pi*np.dot(W,np.sum(B*dl_circle,axis=1)))
    check('有限段直接绕圆求积',circulation(Bs[0]),h/math.sqrt(r*r+h*h))
    check('实际回流三边补足环流',circulation(sum(Bs[1:])),1-h/math.sqrt(r*r+h*h))
    check('完整回路的安培环流',circulation(sum(Bs)),1)
report={'lecture':'第九讲','passed':len(tests),'all_passed':True,'checks':tests}
report['groups']={k:{'count':sum(t['kind']==k for t in tests),'max_normalized_error':max(t['normalized_error'] for t in tests if t['kind']==k)} for k in sorted(set(t['kind'] for t in tests))}
(ROOT/'检查/第九讲_公式核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))
