from pathlib import Path
import json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
nodes,weights=np.polynomial.legendre.leggauss(180)
def quad(f,a,b):
 xs=(nodes+1)*(b-a)/2+a
 return float(np.dot(weights,np.array([f(x) for x in xs]))*(b-a)/2)
records=[]
def ck(group,name,got,want,tol=3e-7):
 err=abs(got-want)/max(1,abs(got),abs(want))
 assert err<tol,(name,got,want,err,tol)
 records.append(dict(group=group,name=name,got=float(got),expected=float(want),normalized_error=float(err)))
def deriv(f,x,h=1e-5):return (f(x+h)-f(x-h))/(2*h)
def solve(fn,target):
 lo=0.;hi=1.
 while fn(hi)<target:hi*=2
 for _ in range(90):
  mid=(lo+hi)/2
  if fn(mid)<target:lo=mid
  else:hi=mid
 return (lo+hi)/2
# 空间积分与磁荷场；以下以mu0=1归一化。
for a,M in [(.5,.7),(1.,1.),(2.,.2)]:
 for zratio in [.15,.5,.8,1.5,3.]:
  z=a*zratio
  hz=M*a*a/2*quad(lambda u:u*(z-a*u)/(z*z+a*a-2*z*a*u)**1.5,-1,1)
  wanted=-M/3 if z<a else 2*M*a**3/(3*z**3)
  ck('空间积分',f'sphere-H-{a}-{zratio}',hz,wanted)
 v=4*math.pi*a**3/3
 angular=2*math.pi*quad(lambda u:1+3*u*u,-1,1)
 # 用s=a/r，将无限径向积分变为有限区间。
 out=.5*(M*a**3/3)**2*angular*quad(lambda s:s*s/a**3,0,1)
 ck('空间积分',f'sphere-external-energy-{a}',out,M*M*v/9)
for R,L,M in [(1,.2,.8),(.4,2,1),(2,3,.2)]:
 s=L/2
 disk=M/2*quad(lambda r:s*r/(s*s+r*r)**1.5,0,R)
 nc=1-L/math.sqrt(L*L+4*R*R)
 ck('空间积分',f'cylinder-center-{R}-{L}',-2*disk,-M*nc)
# 固定磁化总荷与偶极矩：球表面一次角分布。
for M,a in [(.3,2.),(1.,.7)]:
 dip=2*math.pi*a**3*M*quad(lambda u:u*u,-1,1)
 ck('空间积分',f'magnet-charge-moment-{a}',dip,4*math.pi*a**3*M/3)
for chi in [-.8,-.1,.2,3.,30.]:
 for z in [.15,.6,1.5,3.]:
  H=lambda zz:1/(2*(1+zz*zz)**1.5)
  U=lambda zz:-.5*chi/(1+chi/3)*H(zz)**2
  F=-3*chi/(1+chi/3)*z/(4*(1+z*z)**4)
  ck('有限差分',f'induced-force-{chi}-{z}',-deriv(U,z),F)
for chi in [-.5,.2,2.]:
 for th in [.2,.7,1.2]:
  np1,np2=.1,.45
  ap=chi/(1+np1*chi);at=chi/(1+np2*chi)
  U=lambda t:-.5*(ap*math.cos(t)**2+at*math.sin(t)**2)
  ck('有限差分',f'ellipsoid-torque-{chi}-{th}',-deriv(U,th),-(ap-at)*math.sin(th)*math.cos(th))
# 球壳边界的4x4线性系统，与消元公式独立比较。
for k in [.05,1,2,10,1000]:
 for ratio in [.2,.7,.98]:
  a=ratio;b=1.
  mat=np.array([[1,-1,-1/a**3,0],[1,-k,2*k/a**3,0],[0,1,1/b**3,1/b**3],[0,k,-2*k/b**3,-2/b**3]],float)
  h,p,q,c=np.linalg.solve(mat,np.array([0,0,1,1],float))
  wanted=9*k/((2*k+1)*(k+2)-2*(k-1)**2*(a/b)**3)
  ck('边界矩阵',f'shield-cavity-{k}-{ratio}',h,wanted)
  ck('边界矩阵',f'shield-outer-normal-{k}-{ratio}',k*(p-2*q/b**3),1+2*c/b**3)
# 不含自由电流的各向异性界面，直接求张量分量。
mu=np.array([[5,0,3],[0,2,0],[3,0,5]],float)
hh=np.array([1,0,-.6]);bb=mu@hh
ck('边界矩阵','anisotropic-normal',bb[2],0)
ck('边界矩阵','anisotropic-energy',hh@bb/2,1.6)
for f,m1,m2 in [(.2,1,4),(.7,.3,20),(.4,2,2)]:
 mt=f*m1+(1-f)*m2;mn=1/(f/m1+(1-f)/m2)
 ck('边界矩阵',f'laminate-difference-{f}',mt-mn,f*(1-f)*(m1-m2)**2/(f*m2+(1-f)*m1))
 ck('边界矩阵',f'laminate-normal-energy-{f}',(f/m1+(1-f)/m2)/2,1/(2*mn))
for e1,e2,g1,g2,d1,d2 in [(2,3,1,4,.5,.8),(1,4,3,.2,1,2),(2,4,1,2,.3,.7)]:
 ae=e1*d2+e2*d1;ag=g1*d2+g2*d1;ta=ae/ag;ss=(g1*e2-g2*e1)/ag
 for tt in [.2*ta,ta,3*ta]:
  sig=lambda t:ss*(1-math.exp(-t/ta))
  E1=lambda t:(e2-d2*sig(t))/ae;E2=lambda t:(e1+d1*sig(t))/ae
  ck('有限差分',f'layer-charge-current-{g1}-{tt}',deriv(sig,tt,h=ta*1e-5),g1*E1(tt)-g2*E2(tt))
  ck('边界矩阵',f'layer-voltage-{g1}-{tt}',d1*E1(tt)+d2*E2(tt),1.)
  ck('有限差分',f'layer-total-current-{g1}-{tt}',g1*E1(tt)+e1*deriv(E1,tt,h=ta*1e-5),g2*E2(tt)+e2*deriv(E2,tt,h=ta*1e-5))
# 非线性电介质能量及微分电容。
for ep,eta,E in [(.7,.3,.5),(1,2,1),(3,.2,2.5)]:
 D=lambda e:ep*e+eta*e**3
 w=quad(lambda e:e*(ep+3*eta*e*e),0,E)
 wc=quad(D,0,E)
 ck('能量积分',f'dielectric-energy-{E}',w,ep*E**2/2+3*eta*E**4/4)
 ck('能量积分',f'dielectric-legendre-{E}',w+wc,E*D(E))
 ck('有限差分',f'dielectric-capacitance-{E}',deriv(D,E),ep+3*eta*E**2)
for a,b,Q in [(.3,1,2),(1,2,.5),(2,2.3,1)]:
 E=lambda r:Q*a/(4*math.pi*r**3)
 u=quad(E,a,b);ww=quad(lambda r:.5*E(r)*(Q/(4*math.pi*r*r))*4*math.pi*r*r,a,b)
 ck('空间积分',f'graded-capacitance-{a}',Q/u,8*math.pi*a*b*b/(b*b-a*a))
 ck('空间积分',f'graded-energy-{a}',ww,Q*u/2)
# 任意径向电流与有限回流层的能量。
for p in [0,.5,2,10,50]:
 a=.7;I=1.2;mi=.8
 enc=lambda r:I*(r/a)**(p+2)
 w=quad(lambda r:mi*enc(r)**2/(4*math.pi*r),0,a)
 ck('空间积分',f'power-current-internal-{p}',2*w/I**2,mi/(4*math.pi*(p+2)))
for b,c in [(.5,1),(1,1.5),(2,3),(1,1.02)]:
 bracket=c**4*math.log(c/b)-c*c*(c*c-b*b)+(c**4-b**4)/4
 exact=bracket/(2*math.pi*(c*c-b*b)**2)
 integ=quad(lambda r:(c*c-r*r)**2/r,b,c)/(2*math.pi*(c*c-b*b)**2)
 ck('空间积分',f'outer-return-inductance-{b}-{c}',integ,exact)
# 非线性单磁路：数值积分I dPsi及恒流余能的数值求导。
for ac,bc,L,S,N,g,I in [(.4,.3,1,2,3,.2,1),(2,.2,.7,1,4,1,2),(.2,3,2,.5,2,.3,.8)]:
 def val(gap):
  A=ac*L+gap;D=bc*L
  B=solve(lambda b:A*b+D*b**3,N*I)
  W=quad(lambda b:S*(A*b+D*b**3),0,B)
  WC=I*N*S*B-W
  return B,W,WC
 B,W,WC=val(g)
 A=ac*L+g;D=bc*L
 ck('能量积分',f'nonlinear-core-energy-{g}',W,S*(A*B*B/2+D*B**4/4))
 ck('有限差分',f'nonlinear-core-force-{g}',deriv(lambda x:val(x)[2],g),-S*B*B/2)
# 分支非线性磁路，直接从充磁功积分得到余能，再改变x重解。
for ac,bc,L,Sc,S1,S2,g,N,I in [(1,.4,.8,1,.5,.7,1,3,1),(.2,2,1.2,.8,1,.5,.7,2,.5),(2,.1,.5,2,.4,.6,2,4,2)]:
 for xx in [.3,.8,1.7]:
  def value(x,cur=I):
   P=S1/x+S2/g
   fn=lambda V:V+L*(ac*P*V/Sc+bc*(P*V/Sc)**3)
   V=solve(fn,N*cur);Bc=P*V/Sc
   W=quad(lambda b:Sc*L*(ac*b+bc*b**3),0,Bc)+P*V*V/2
   WC=cur*N*P*V-W
   return P,V,Bc,W,WC
  P,V,Bc,W,WC=value(xx)
  ck('有限差分',f'branch-force-{ac}-{xx}',deriv(lambda x:value(x)[4],xx),-S1*V*V/(2*xx*xx),2e-6)
  diff=deriv(lambda ii:N*value(xx,ii)[0]*value(xx,ii)[1],I)
  want=N*N*P/(1+L*P/Sc*(ac+3*bc*Bc*Bc))
  ck('有限差分',f'branch-diff-inductance-{ac}-{xx}',diff,want)
  ck('能量积分',f'branch-coenergy-{ac}-{xx}',WC,L*Sc*(ac*Bc*Bc/2+3*bc*Bc**4/4)+P*V*V/2)
# 耦合线圈约束与弹簧失稳的函数导数。
for x in [.3,.7,1.4]:
 M=lambda xx:.5*math.exp(-xx)
 le=lambda xx:1-M(xx)**2/2
 ck('有限差分',f'secondary-force-{x}',deriv(lambda xx:le(xx)/2,x),M(x)**2/2)
for d0,delta in [(1,0),(1,.2),(1,1.9),(1,2),(1,3)]:
 D=d0+delta;xc=D/3;K=4*D**3/27
 pi=lambda x:.5*x*x-K/(D-x)
 ck('有限差分',f'pullin-first-{delta}',deriv(pi,xc),0)
 h=1e-4
 ck('有限差分',f'pullin-second-{delta}',(pi(xc+h)-2*pi(xc)+pi(xc-h))/h**2,0,3e-7)
 ck('代数关系',f'pullin-equilibrium-{delta}',xc*(D-xc)**2,K)
 if delta>=2:
  contactK=d0*delta**2
  ck('代数关系',f'contact-equilibrium-{delta}',contactK/(D-d0)**2,d0)
for a,L,g,I in [(.7,2,3,1),(1,.5,2,3)]:
 Ez=I/(g*math.pi*a*a);Hp=I/(2*math.pi*a)
 pin=quad(lambda ph:Ez*Hp*a*L,0,2*math.pi)
 ck('空间积分',f'resistor-power-{a}',pin,I*I*L/(g*math.pi*a*a))
# 正式稿中的舍入值独立核对。
mu0=4*math.pi*1e-7
B=mu0*250/(.001+.2/2000);LL=mu0*500**2*4e-4/.0011
for name,got,want in [('B-W7',B,.2856),('L-W7',LL,.1142),('energy-W7',LL*.5**2/2,.01428),('force-W7',4e-4*B*B/(2*mu0),12.98)]:
 ck('题给数值',name,got,want,8e-4)
k=1000;f=1/1.02**3;sf=1+2*(k-1)**2*(1-f)/(9*k)
ck('题给数值','shield-factor',sf,13.79,8e-4)
ck('题给数值','shield-ratio',1/sf,.07251,8e-5)
ck('题给数值','shield-thin',1+2*(k-1)**2*.02/(3*k),14.31,5e-4)
h=4e5;bstar=1.2-mu0*1.05*h;pm=bstar*h;wg=.6**2*1e-6/(2*mu0);vm=2*wg/pm
for name,got,want in [('BHmax',pm,2.689e5),('magnet-B',bstar,.6722),('gap-energy',wg,.1432),('magnet-volume-cm3',vm*1e6,1.065),('magnet-area-cm2',.6*5/bstar,4.463),('magnet-length-mm',.6*.002/(mu0*h)*1000,2.387)]:
 ck('题给数值',name,got,want,8e-4)
groups={}
for r in records:
 gr=groups.setdefault(r['group'],dict(count=0,maximum_normalized_error=0))
 gr['count']+=1;gr['maximum_normalized_error']=max(gr['maximum_normalized_error'],r['normalized_error'])
result=dict(count=len(records),all_passed=True,count_description='不同参数下的空间积分、边界矩阵、能量积分、有限差分与舍入比较实例，非独立公式数量',normalized_error_definition='abs(got-expected)/max(1,abs(got),abs(expected));零值用单位化绝对误差',groups=groups,records=records)
(ROOT/'检查/第十八讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
print('precise rounded source values',dict(shield=sf,ratio=1/sf,thin=1+2*(k-1)**2*.02/(3*k),Bstar=bstar,Pmax=pm,Wgap=wg,Vmin_cm3=vm*1e6))
