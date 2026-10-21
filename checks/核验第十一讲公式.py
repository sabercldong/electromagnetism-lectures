from pathlib import Path
import json,math
import numpy as np
_nodes,_weights=np.polynomial.legendre.leggauss(160)
def quad(f,a,b,points=None):
    if b<a:
        val,err=quad(f,b,a,points=points);return -val,err
    cuts=[a]+sorted(x for x in (points or []) if a<x<b)+[b]
    val=0.
    for left,right in zip(cuts,cuts[1:]):
        mid=(left+right)/2;half=(right-left)/2
        val+=half*sum(w*f(mid+half*t) for t,w in zip(_nodes,_weights))
    return val,0.
def eigh_tridiagonal(diag,off,select,select_range,eigvals_only):
    # Sturm计数与二分求低阶本征值，避免构造稠密矩阵。
    radius=np.r_[0.,np.abs(off)]+np.r_[np.abs(off),0.]
    lo=float(np.min(diag-radius))-1.;hi=float(np.max(diag+radius))+1.
    off2=off*off
    def count(x):
        pivot=diag[0]-x;n=int(pivot<0)
        for k in range(1,len(diag)):
            if abs(pivot)<1e-100:pivot=-1e-100
            pivot=diag[k]-x-off2[k-1]/pivot;n+=int(pivot<0)
        return n
    vals=[]
    for j in range(select_range[0],select_range[1]+1):
        left,right=lo,hi
        for _ in range(65):
            mid=(left+right)/2
            if count(mid)<=j:left=mid
            else:right=mid
        vals.append((left+right)/2)
    return np.array(vals)
P=Path(__file__).resolve().parent
checks=[]
def ck(name,actual,expected,tol=2e-8,kind='积分或恒等式'):
    a=np.asarray(actual,float);b=np.asarray(expected,float)
    err=float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(b)))))
    checks.append(dict(name=name,normalized_error=err,tolerance=tol,kind=kind))
    assert err<tol,(name,err,tol)
def d(f,x,h=1e-5):return (f(x+h)-f(x-h))/(2*h)
def d2(f,x,h=1e-4):return (f(x+h)-2*f(x)+f(x-h))/h**2
def rk(f,y,tmax,dt=.002):
    y=np.array(y,float);n=math.ceil(tmax/dt);h=tmax/n
    for j in range(n):
        t=j*h;k1=f(t,y);k2=f(t+h/2,y+h*k1/2);k3=f(t+h/2,y+h*k2/2);k4=f(t+h,y+h*k3)
        y+=h*(k1+2*k2+2*k3+k4)/6
    return y
# 直接洛伦兹方程积分，独立于解析轨迹表达式。
for charge in [-1.3,.6,1.7]:
 for ratio in [.3,1.2,2.5]:
  B=.9;m=1.1;E=ratio*B;omega=charge*B/m
  for vx0,vy0 in [(0,0),(ratio+.07,0),(ratio,.8)]:
   for t in [.8,2.3,6.1]:
    def f(t,s):
     x,y,vx,vy=s;return np.array([vx,vy,charge*B*vy/m,charge*(E-B*vx)/m])
    s=rk(f,[0,0,vx0,vy0],t)
    w=vx0-ratio;sn=math.sin(omega*t);co=math.cos(omega*t)
    exact=[ratio*t+w*sn/omega+vy0*(1-co)/omega,w*(co-1)/omega+vy0*sn/omega,ratio+w*co+vy0*sn,vy0*co-w*sn]
    ck('正交场轨迹 q=%g vD=%g t=%g'%(charge,ratio,t),s,exact,3e-9,'运动方程积分')
    ck('电场功与动能差',m*(s[2]**2+s[3]**2-vx0**2-vy0**2)/2,charge*E*s[1],3e-9,'运动方程积分')
# 点源会聚：任意初始横向速度，整周期回到轴线。
for charge in [-.8,1.2]:
 for vxy in [(.4,0),(0,.7),(.3,-.9)]:
  def f(t,s):return np.array([s[2],s[3],charge*s[3],-charge*s[2]])
  s=rk(f,[0,0,*vxy],2*np.pi/abs(charge));ck('整周期磁聚焦',s[:2],[0,0],5e-10,'运动方程积分')
  s=rk(f,[0,0,*vxy],np.pi/abs(charge));ck('半周期最大离轴距离',np.linalg.norm(s[:2]),2*np.linalg.norm(vxy)/abs(charge),5e-10,'运动方程积分')
# Drude逐种粒子解线性受力方程，构造电导张量，再反演。
def conductivity(n,p,me,mh,B):
 def one(density,q,mu):
  mat=np.array([[1/mu,-q*B],[q*B,1/mu]])
  return density*q*np.linalg.solve(mat,q*np.eye(2))
 return one(n,-1,me)+one(p,1,mh)
for B in [0,.2,1.5,4.]:
 for me,mh in [(.3,.3),(.2,.7),(.9,.25)]:
  n=1.7;rho=np.linalg.inv(conductivity(n,n,me,mh,B))
  expected=np.array([[(1+me*mh*B*B)/(n*(me+mh)),-B*(mh-me)/(n*(me+mh))],[B*(mh-me)/(n*(me+mh)),(1+me*mh*B*B)/(n*(me+mh))]])
  ck('补偿双载流子精确电阻张量',rho,expected)
for q in [-1,1]:
 for mu in [.2,.8]:
  for B in [.1,1.,3.]:
   mat=np.array([[1/mu,-q*B],[q*B,1/mu]])
   sig=q*np.linalg.solve(mat,q*np.eye(2));rho=np.linalg.inv(sig)
   ck('单载流子横向开路',rho[0,0],1/mu)
   ck('单载流子横向短路',1/sig[0,0],(1+(mu*B)**2)/mu)
# 自旋电阻网络：给定左右电势，求两中间节点及端口电流。
for r,R in [(.3,1.2),(1,2),(2,2),(1,20)]:
 def resistance(l1,r1,l2,r2):
  v1=(1/l1)/(1/l1+1/r1);v2=(1/l2)/(1/l2+1/r2)
  return 1/((1-v1)/l1+(1-v2)/l2)
 rp=resistance(r,r,R,R);ra=resistance(r,R,R,r)
 ck('两自旋通道相对磁阻',(ra-rp)/rp,(R-r)**2/(4*r*R))
# 无量纲圆柱体电流：由源积分得到B，与势的有限差分独立比较。
for taper in [False,True]:
 R=1.3;J=lambda r:1-r*r/R**2 if taper else 1.
 total=quad(lambda r:J(r)*r,0,R)[0]
 def B(r):return quad(lambda s:J(s)*s,0,min(r,R))[0]/r
 def A(r):
  if r>=R:return -total*math.log(r/R)
  if taper:return 3*R*R/16-r*r/4+r**4/(16*R*R)
  return (R*R-r*r)/4
 for r in [.15,.45,.9,1.2,1.5,2.7]:
  ck('圆柱势的旋度',-d(A,r),B(r),2e-9,'有限差分')
  ck('圆柱势的径向磁通积分',A(r),quad(B,r,R)[0])
  if r<R:ck('圆柱泊松方程',d2(A,r)+d(A,r)/r,-J(r),3e-8,'有限差分')
# 同轴电流与矢势；各区积分及粒子正则动量。
a,b,c=1.,1.6,2.2;C=.4;D=c*c-b*b
F=lambda r:C/D*(c*c*math.log(c/r)-(c*c-r*r)/2)
Ab=F(b);Aa=Ab+C*math.log(b/a)
def A(r):
 if r<a:return Aa+C*(1-r*r/a**2)/2
 if r<b:return Ab+C*math.log(b/r)
 if r<c:return F(r)
 return 0.
def Bphi(r):
 if r<a:return C*r/a**2
 if r<b:return C/r
 if r<c:return C*(c*c/r-r)/D
 return 0.
for r in [.2,.5,.8,1.2,1.8,2.,2.5]:
 val=quad(Bphi,r,max(c,r),points=[s for s in [a,b] if r<s<c])[0] if r<c else 0
 ck('同轴矢势与直接分段场积分',A(r),val)
 ck('同轴矢势求旋度',-d(A,r),Bphi(r),3e-9,'有限差分')
for r0 in [a,b,c]:
 ck('同轴界面矢势连续',A(r0-1e-7),A(r0+1e-7),1e-7)
 ck('同轴界面磁场连续',Bphi(r0-1e-7),Bphi(r0+1e-7),2e-7)
for vr0 in [1.4,2.,3.]:
 q=.7;vz0=.9;A0=A(.5)
 def lor(t,s):
  pos=s[:3];vel=s[3:];r=np.linalg.norm(pos[:2]);B=Bphi(r)*np.array([-pos[1]/r,pos[0]/r,0.])
  return np.r_[vel,q*np.cross(vel,B)]
 for t in [.5,1.,2.5]:
  s=rk(lor,[.5,0,0,vr0,0,vz0],t,.0005);r=np.linalg.norm(s[:2])
  ck('同轴完整洛伦兹轨迹的正则轴向动量',s[5]+q*A(r),vz0+q*A0,2e-8,'运动方程积分')
  ck('同轴完整洛伦兹轨迹的机械能',np.dot(s[3:],s[3:]),vr0**2+vz0**2,2e-8,'运动方程积分')
for delta in [1e-2,3e-3,1e-3]:
 ck('同轴外边界二次势的截断',A(c-delta),C/D*delta**2,2e-6,'渐近比较')
# 静态轴对称边缘场，完整三维洛伦兹积分检验角动量。
for q in [-1.,.8]:
 def field(pos):
  x,y,z=pos;v=math.tanh(z);bz=(1+v)/2;dp=(1-v*v)/2
  return np.array([-x*dp/2,-y*dp/2,bz])
 def lor(t,s):return np.r_[s[3:],q*np.cross(s[3:],field(s[:3]))]
 init=np.array([.25,0,-4,0,0,1.5]);l0=q*(1+math.tanh(-4))*.25**2/4
 for t in [2.,4.,6.]:
  s=rk(lor,init,t,.001);x,y,z=s[:3];vx,vy,vz=s[3:];bz=field(s[:3])[2]
  ck('边缘场正则角动量',x*vy-y*vx+q*bz*(x*x+y*y)/2,l0,2e-10,'运动方程积分')
# 规范、圆周积分以及直线对的旋度。
for R in [.4,1,2.3]:
 ck('对称与朗道规范闭合环流',quad(lambda t:R*R*math.cos(t)**2,0,2*np.pi)[0],np.pi*R*R)
for x,y in [(.3,.7),(-.5,.4),(0,1.2),(2.,1.3)]:
 arad=.8
 az=lambda xx,yy:math.log(math.hypot(xx-arad,yy)/math.hypot(xx+arad,yy))
 curl=np.array([d(lambda yy:az(x,yy),y),-d(lambda xx:az(xx,y),x)])
 direct=sum((sg*np.array([-y,x-x0])/((x-x0)**2+y*y) for x0,sg in [(-arad,1),(arad,-1)]),np.zeros(2))
 ck('反向双导线势的旋度与直线毕奥场',curl,direct,2e-9,'有限差分')
for h in [.2,.7,1.6]:
 for R in [.3,1.,3.]:
  flux=quad(lambda r:r*(2*h*h-r*r)/(r*r+h*h)**2.5,0,R)[0]
  ck('偶极子圆盘磁通与圆周矢势',flux,R*R/(R*R+h*h)**1.5)
# 开口电流积分的散度有限差分，与端点表达式比对。
def av(pos,closed=False):
 if closed:
  return np.array([quad(lambda t:comp(t)/np.linalg.norm(pos-np.array([math.cos(t),math.sin(t),0])),0,2*np.pi)[0] for comp in [lambda t:-math.sin(t),lambda t:math.cos(t),lambda t:0.]])
 return np.array([quad(lambda u:comp(u)/np.linalg.norm(pos-np.array([u,.3*math.sin(np.pi*u),0])), -1,1)[0] for comp in [lambda u:1,lambda u:.3*np.pi*math.cos(np.pi*u),lambda u:0]])
for pos in [np.array([.2,.8,1.]),np.array([2.,.3,.6])]:
 for closed in [False,True]:
  eps=1e-4;div=sum((av(pos+eps*np.eye(3)[i],closed)[i]-av(pos-eps*np.eye(3)[i],closed)[i])/(2*eps) for i in range(3))
  expected=0 if closed else 1/np.linalg.norm(pos-[-1,0,0])-1/np.linalg.norm(pos-[1,0,0])
  ck('开口与闭合电流矢势散度',div,expected,2e-8,'有限差分')
for R in [1.1,2.,5.,20.]:
 alpha=math.asin(1/math.sqrt(R))
 ck('双损失锥粒子数积分',quad(lambda t:math.sin(t),0,alpha)[0],1-math.sqrt(1-1/R))
 ck('单端损失锥通量积分',quad(lambda t:2*math.sin(t)*math.cos(t),0,alpha)[0],1/R)
# 谐振子薛定谔方程的网格本征值，独立核对朗道轨道间隔。
x=np.linspace(-12,12,1201);dx=x[1]-x[0]
for omega in [.5,1.,2.]:
 diag=1/dx**2+.5*omega**2*x[1:-1]**2;off=np.full(len(diag)-1,-.5/dx**2)
 eig=eigh_tridiagonal(diag,off,select='i',select_range=(0,2),eigvals_only=True)
 ck('朗道对应谐振子的数值本征值',eig,omega*(np.arange(3)+.5),1.5e-4,'量子网格近似')
e=1.602176634e-19;h=6.62607015e-34;me=9.1093837139e-31;hb=h/(2*np.pi)
ck('例12霍尔电压',h/(2*e*e)*2e-6,.025813,2e-7,'数值代入与有效位数')
ck('例12面密度',2*e*5/h/1e15,2.4180,3e-5,'数值代入与有效位数')
ck('例12磁长度纳米',math.sqrt(hb/(e*5))*1e9,11.47,.0006,'数值代入与有效位数')
ck('例12轨道间隔毫电子伏',hb*e*5/me/e*1e3,.579,.0002,'数值代入与有效位数')
B=3e15*h/(3*e)
ck('题16整数填充磁场',B,4.1357,2e-5,'数值代入与有效位数')
ck('题16有效质量轨道间隔',hb*e*B/(.067*me)/e*1e3,7.15,.001,'数值代入与有效位数')
groups={}
for c0 in checks:
 g=groups.setdefault(c0['kind'],dict(count=0,maximum_normalized_error=0.))
 g['count']+=1;g['maximum_normalized_error']=max(g['maximum_normalized_error'],c0['normalized_error'])
result=dict(lecture='第十一讲',passed=len(checks),all_passed=True,groups=groups,checks=checks)
(P/'检查/第十一讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
