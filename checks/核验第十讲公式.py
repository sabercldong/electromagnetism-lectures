"""以源积分、输运方程和运动方程核验第十讲。力学计算取 mu0=1。"""
from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parent
X,W=leggauss(160);tests=[]
def quad(f,a,b):
    t=(a+b)/2+(b-a)*X/2
    return (b-a)/2*np.tensordot(W,np.asarray(f(t)),axes=(0,0))
def check(name,v,e,tol=2e-9,kind='积分或恒等式'):
    v,e=np.asarray(v),np.asarray(e)
    err=float(np.max(np.abs(v-e))/max(1,float(np.max(np.abs(e)))))
    assert err<tol,(name,err,tol,v,e)
    tests.append(dict(name=name,normalized_error=err,tolerance=tol,kind=kind))
for R,J in [(.7,1.3),(1,2),(2,.8)]:
    for mode in ['uniform','center']:
        j=(lambda r:J+0*r) if mode=='uniform' else (lambda r:J*(1-r*r/R**2))
        I=quad(lambda r:2*math.pi*j(r)*r,0,R)
        check('圆柱总电流 '+mode,I,math.pi*J*R**2/(1 if mode=='uniform' else 2))
        for r in [.15*R,.4*R,.85*R,R,1.4*R]:
            B=quad(lambda s:j(s)*s,0,min(r,R))/r
            inside=J*r/2 if mode=='uniform' else J*r/2*(1-r*r/(2*R*R))
            check('累计源电流磁场 '+mode,B,inside if r<=R else I/(2*math.pi*r))
        for r in [0,.2*R,.6*R,R]:
            b=lambda s:J*s/2 if mode=='uniform' else J*s/2*(1-s*s/(2*R*R))
            p=quad(lambda s:j(s)*b(s),r,R)
            expected=J*J*(R*R-r*r)/4 if mode=='uniform' else J*J*(5*R*R/48-r*r/4+3*r**4/(16*R*R)-r**6/(24*R**4))
            check('磁挤压压力源积分 '+mode,p,expected)
            if 0<r<R:
                h=1e-5
                pf=lambda r:J*J*(R*R-r*r)/4 if mode=='uniform' else J*J*(5*R*R/48-r*r/4+3*r**4/(16*R*R)-r**6/(24*R**4))
                check('压力梯度与磁力',(pf(r+h)-pf(r-h))/(2*h),-j(r)*b(r),tol=3e-9,kind='差分')
for a,b,h in [(.7,1.3,.4),(1,1.1,2),(1,3,1.5)]:
    check('环形截面径向磁通',h*quad(lambda r:1/(2*math.pi*r),a,b),h*math.log(b/a)/(2*math.pi))
for t in [.03,.06,.1]:
    exact=math.log((1+t/2)/(1-t/2));approx=t+t**3/12
    check('窄环磁通截断',exact,approx,tol=2e-7,kind='渐近截断')
for R,B in [(.5,2),(1,1.3),(2,.4)]:
    check('半球有向磁通',2*math.pi*quad(lambda theta:B*R*R*np.cos(theta)*np.sin(theta),0,math.pi/2),math.pi*B*R*R)
    for g in [-.7,0,.9]:
        force=quad(lambda t:np.column_stack((R*np.cos(t)*(B+g*R*np.cos(t)),R*np.sin(t)*(B+g*R*np.cos(t)),0*t)),math.pi,0)
        check('半圆弧力直接积分',force,[-math.pi*g*R*R/2,-2*B*R,0])
        diameter=quad(lambda x:np.column_stack((0*x,-(B+g*x),0*x)),R,-R)
        check('直径回流力直接积分',diameter,[0,2*B*R,0])
        check('半圆闭合回路梯度力',force+diameter,[-math.pi*g*R*R/2,0,0])
for a,b,ell in [(.4,1.4,.8),(1,1.3,2),(1,4,.5)]:
    near=np.cross([0,ell,0],[0,0,-1/(2*math.pi*a)])
    far=np.cross([0,-ell,0],[0,0,-1/(2*math.pi*b)])
    upper=quad(lambda x:np.column_stack((0*x,1/(2*math.pi*x),0*x)),a,b)
    lower=quad(lambda x:np.column_stack((0*x,1/(2*math.pi*x),0*x)),b,a)
    check('矩形回路四边积分',near+far+upper+lower,[-ell/(2*math.pi)*(1/a-1/b),0,0])
# 对不同朝向的闭合椭圆，直接积分安培力和力矩。
for tilt in [.2,.8,1.4]:
    u=np.array([math.cos(tilt),0,math.sin(tilt)]);v=np.array([0,1,0])
    for aa,bb in [(1,.7),(.4,.9)]:
        moment=math.pi*aa*bb*np.cross(u,v)
        def geometry(t):
            return aa*np.cos(t)[:,None]*u+bb*np.sin(t)[:,None]*v,-aa*np.sin(t)[:,None]*u+bb*np.cos(t)[:,None]*v
        for G in [np.diag([-.4,-.4,.8]),np.array([[0,0,0],[0,0,0],[.7,0,0]])]:
            B0=np.array([.3,-.2,1.1])
            def f(t):
                pos,dl=geometry(t);return np.cross(dl,B0+pos@G.T)
            check('线性场闭合椭圆合力',quad(f,0,2*math.pi),G.T@moment)
            check('均匀场线圈力矩',quad(lambda t:np.cross(geometry(t)[0],np.cross(geometry(t)[1],B0)),0,2*math.pi),np.cross(moment,B0))
for alpha,beta in [(.3,-.7),(1,2),(-.5,.5)]:
    gamma=-alpha-beta;a=1.7
    flux=8*a**3*(alpha+beta+gamma)
    check('线性场六面通量',flux,0)
for a in [.6,1,1.8]:
    def wirefield(x,y):
        rv=np.array([x-a,y]);lv=np.array([x+a,y])
        return (np.array([-rv[1],rv[0]])/(rv@rv)+np.array([-lv[1],lv[0]])/(lv@lv))/(2*math.pi)
    h=1e-6
    jac=np.column_stack(((wirefield(h,0)-wirefield(-h,0))/(2*h),(wirefield(0,h)-wirefield(0,-h))/(2*h)))
    check('三导线横向场梯度',jac,[[0,-1/(math.pi*a*a)],[-1/(math.pi*a*a),0]],tol=3e-9,kind='差分')
for theta in [.2,math.pi/3,1.7]:
    check('转动力矩的机械功',quad(lambda t:-np.sin(t),theta,0),1-math.cos(theta))
for phi in [.01,.02,.04]:
    check('临界扭转四阶势能',phi*phi/2+math.cos(phi)-1,phi**4/24,tol=7e-12,kind='渐近截断')
# 按带符号载流子漂移方程求输运张量，与弱场 Hall 系数比较。
def conductivity(carriers,B):
    C=np.array([[0,B],[-B,0]])
    return sum(density*mob*np.linalg.inv(np.eye(2)-sign*mob*C) for density,mob,sign in carriers)
for sign in [-1,1]:
    for B in [.01,.7,2]:
        n,mob=2.4,.8;S=conductivity([(n,mob,sign)],B)
        Ey=-S[1,0]/S[1,1];jx=S[0,0]+S[0,1]*Ey
        check('单载流子 Hall 符号与系数',Ey/(jx*B),1/(n*sign))
for n,p,me,mh in [(3,1,.5,1),(4,1,.7,1.4),(1,3,1,.4),(2,.5,1,2)]:
    B=1e-4;S=conductivity([(n,me,-1),(p,mh,1)],B)
    Ey=-S[1,0]/S[1,1];jx=S[0,0]+S[0,1]*Ey
    expected=(p*mh*mh-n*me*me)/(p*mh+n*me)**2
    check('双载流子弱场输运方程',Ey/(jx*B),expected,tol=5e-8,kind='弱场近似')
U=np.array([7.5,3.1,-6.5,-2.9]);M=np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]])
check('反转读数的四分量',M@U/4,[.3,5,2,.2])
check('反转分量还原读数',M.T@np.array([.3,5,2,.2]),U)
for a,b,ell,d,n,sign in [(.4,1,.8,.2,3,1),(1,2,1.3,.1,2,-1),(2,2.2,2,.3,5,1)]:
    w=b-a;C=1/(2*math.pi);jy=1/(w*d)
    U=quad(lambda x:C*jy/(n*sign*x),a,b)
    Fx=d*ell*quad(lambda x:-C*jy/x,a,b)
    check('非均匀 Hall 横向电压积分',U,C/(n*sign*w*d)*math.log(b/a))
    check('非均匀 Hall 条磁力积分',Fx,-C*ell/w*math.log(b/a))
    check('Hall 电压与支承力消参',Fx,-n*sign*d*ell*U)
e=1.602176634e-19
check('例题 Hall 浓度',.005*.8/(e*.0002*.0025),4.99320725956861e22,tol=2e-9)
check('例题荷质比',2000/(.2**2*(.02**2+.02**2)),6.25e7)
check('例题迁移率',(.005*.02/(.003*.0002*.4))*.000125,.0520833333333333)

def rk4(f,t,state,h):
    a=f(t,state);b=f(t+h/2,state+h*a/2);c=f(t+h/2,state+h*b/2);d=f(t+h,state+h*c)
    return state+h*(a+2*b+2*c+d)/6
def constant_rhs(q,m,B,E):
    return lambda t,s:np.r_[s[3:],q/m*(np.array([0,0,E])+np.cross(s[3:],[0,0,B]))]
for q,m,B,E in [(1,2,1.3,0),(-1,1.4,.7,0),(1,1,.8,.3),(-1,2,1.1,.2)]:
    omega=q*B/m;T=2*math.pi/abs(omega);initial=np.array([0.,0,0,.7,0,.3]);s=initial.copy();f=constant_rhs(q,m,B,E);dt=T/1600
    for j in range(1,1601):
        s=rk4(f,(j-1)*dt,s,dt)
        if j in [400,800,1600]:
            t=j*dt
            exact=[.7/omega*math.sin(omega*t),.7/omega*(math.cos(omega*t)-1),.3*t+q*E*t*t/(2*m),.7*math.cos(omega*t),-.7*math.sin(omega*t),.3+q*E*t/m]
            check('完整 Lorentz 方程回旋及平行电场',s,exact,tol=3e-9,kind='运动方程积分')
            check('电场功与动能增量',m/2*(s[3:]@s[3:]-initial[3:]@initial[3:]),q*E*s[2],tol=2e-9,kind='运动方程积分')
# 磁场条带的首次出射：积分后对最后一步的事件时刻二分。
for d in [.2,.6,.9,1.3]:
    f=constant_rhs(1,1,1,0);s=np.array([0.,0,0,1,0,0]);t=0;dt=2*math.pi/1600
    while True:
        nxt=rk4(f,t,s,dt)
        right=nxt[0]>=d;left=nxt[0]<=0 and t>.1
        if right or left:break
        s=nxt;t+=dt
    lo,hi=0.,dt;bound=d if right else 0.
    for _ in range(40):
        mid=(lo+hi)/2;tmp=rk4(f,t,s,mid)
        crossed=tmp[0]>=bound if right else tmp[0]<=bound
        if crossed:hi=mid
        else:lo=mid
    end=rk4(f,t,s,(lo+hi)/2);te=t+(lo+hi)/2
    expected=[math.asin(d),d,-1+math.sqrt(1-d*d)] if d<1 else [math.pi,0,-2]
    check('有限磁条带首次出射事件',[te,end[0],end[1]],expected,tol=3e-9,kind='运动方程积分')
# 在无散、无旋的三维磁镜场内积分完整洛伦兹方程，与绝热导引中心比较。
L=2.;Bc=1.;q=100.;m=1.;vp=1.;vz=.4;rho0=m*vp/(q*Bc)
state=np.array([rho0,0,0,0,-vp,vz]);mu0=m*vp*vp/(2*Bc)
def mirror_rhs(t,s):
    x,y,z=s[:3];b=np.array([-Bc*z*x/L**2,-Bc*z*y/L**2,Bc*(1+z*z/L**2-(x*x+y*y)/(2*L**2))])
    return np.r_[s[3:],q/m*np.cross(s[3:],b)]
dt=.0002
for j in range(1,22501):
    state=rk4(mirror_rhs,(j-1)*dt,state,dt)
    if j in [7500,15000,22500]:
        t=j*dt;expected_z=L*vz/vp*math.sin(vp*t/L)
        check('完整三维磁镜与最低阶往返运动',state[2],expected_z,tol=5e-4,kind='绝热近似')
        x,y,z=state[:3];bv=np.array([-Bc*z*x/L**2,-Bc*z*y/L**2,Bc*(1+z*z/L**2-(x*x+y*y)/(2*L**2))]);bm=np.linalg.norm(bv)
        parallel=state[3:]@bv/bm;mu=m*(state[3:]@state[3:]-parallel**2)/(2*bm)
        check('完整三维磁镜的绝热磁矩',mu,mu0,tol=5e-4,kind='绝热近似')
        check('完整三维静磁镜动能',state[3:]@state[3:],vp*vp+vz*vz,tol=2e-7,kind='运动方程积分')
for q in [-1,1]:
    for B,Bprime,vp in [(.8,.3,.7),(1,-.5,1.2),(2,.4,.9)]:
        m=1.4;rho=m*vp/(abs(q)*B);vphi=-q*B*rho/m;Br=-rho*Bprime/2
        force=np.cross([0,vphi,0],[Br,0,B])[2]*q
        check('径向场与带符号回旋的镜力',force,-m*vp*vp/(2*B)*Bprime)
for alpha in [math.radians(35),math.pi/4,1.2]:
    k=1/math.sin(alpha)**2
    check('镜点纵向动能',1-k*math.sin(alpha)**2,0)
for r,H,k in [(.2,.3,1.2),(.5,.5,1.5),(.3,.7,2)]:
    check('磁镜回旋半径缩放',r*math.sqrt(k)/k,r/math.sqrt(k))
    check('综合题入口与镜点能量',k*r*r+H*H-(k-1)*r*r,r*r+H*H)
for p in [.1,.7,2,5]:
    m=1.;c=3.;K=math.sqrt(m*m*c**4+p*p*c*c)-m*c*c;gamma=1+K/(m*c*c)
    check('相对论周期与动量能量',gamma,math.sqrt(1+p*p/(m*m*c*c)))

groups={}
for t in tests:
    g=groups.setdefault(t['kind'],{'count':0,'maximum_normalized_error':0})
    g['count']+=1;g['maximum_normalized_error']=max(g['maximum_normalized_error'],t['normalized_error'])
result={'lecture':'第十讲','passed':len(tests),'all_passed':True,'groups':groups,'checks':tests}
(ROOT/'检查/第十讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
