from pathlib import Path
import json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
nodes,weights=np.polynomial.legendre.leggauss(160)
def quad(f,a,b):
 xx=(nodes+1)*(b-a)/2+a
 return float(np.dot(weights,np.array([f(x) for x in xx]))*(b-a)/2)
records=[]
def check(group,name,got,want,tol=3e-7):
 err=abs(got-want)/max(1.,abs(got),abs(want));assert err<tol,(name,got,want,err)
 records.append({'group':group,'name':name,'got':float(got),'expected':float(want),'normalized_error':float(err)})
# 单位化：mu0=m_a=kB=n=1，检查统计平均、能量导数与回线积分。
for eta in [.001,.05,.3,1.,3.]:
 wp=eta+math.sqrt(1+eta*eta);wm=eta-math.sqrt(1+eta*eta)
 check('代数与统计积分',f'orbit-root-plus-{eta}',wp*wp-2*eta*wp,1.)
 check('代数与统计积分',f'orbit-root-minus-{eta}',wm*wm-2*eta*wm,1.)
 check('代数与统计积分',f'orbit-sum-{eta}',wp+wm,2*eta)
 check('代数与统计积分',f'orbit-kinetic-{eta}',.5*(wp*wp+wm*wm),1+2*eta*eta)
for xi in [.01,.05,.2,.5,1.,2.,5.,10.]:
 zz=quad(lambda u:math.exp(xi*u),-1,1)
 av=quad(lambda u:u*math.exp(xi*u),-1,1)/zz
 check('代数与统计积分',f'continuous-average-{xi}',av,1/math.tanh(xi)-1/xi)
 p=math.exp(xi)/(math.exp(xi)+math.exp(-xi));check('代数与统计积分',f'two-state-{xi}',2*p-1,math.tanh(xi))
 T=1/xi;eps=T*1e-5
 u=lambda t:-math.tanh(1/t)
 check('有限差分',f'two-state-heat-{xi}',(u(T+eps)-u(T-eps))/(2*eps),xi*xi/math.cosh(xi)**2)
 check('代数与统计积分',f'fluid-pressure-{xi}',quad(lambda s:(s/3-s**3/45+2*s**5/945) if abs(s)<1e-3 else 1/math.tanh(s)-1/s,0,xi),math.log(math.sinh(xi)/xi))
check('代数与统计积分','isotropic-u2',quad(lambda u:u*u,-1,1)/2,1/3)
for hm in [.1,.4,1.,2.]:
 for a,b in [(.2,.3),(.7,1.2),(2.,.05)]:
  up=lambda h:(a+b*hm)*h-b/2*(hm*hm-h*h)
  down=lambda h:(a+b*hm)*h+b/2*(hm*hm-h*h)
  check('回线与空间积分',f'Rayleigh-area-{hm}-{b}',quad(lambda h:down(h)-up(h),-hm,hm),4*b*hm**3/3)
  check('回线与空间积分',f'Rayleigh-HdM-{hm}-{b}',quad(lambda h:h*((a+b*hm)+b*h),-hm,hm)+quad(lambda h:h*((a+b*hm)-b*h),hm,-hm),4*b*hm**3/3)
for hp,bp,delta in [(1,2,.05),(2,.4,.3),(.2,3,.8),(1,1,1.5)]:
 w=quad(lambda s:-hp*bp*math.cos(s)*math.sin(s-delta),0,2*math.pi)
 check('回线与空间积分',f'ellipse-area-{delta}',w,math.pi*hp*bp*math.sin(delta))
 for s in [.1,.8,2.,4.]:
  xx=math.cos(s);yy=math.cos(s-delta)
  check('代数与统计积分',f'ellipse-equation-{delta}-{s}',xx*xx+yy*yy-2*xx*yy*math.cos(delta),math.sin(delta)**2)
for d in [.1,.5,2.]:
 for w in [.3,2.,5.]:
  pe=quad(lambda s:quad(lambda x:(x*w*math.cos(s))**2,-d/2,d/2)/d,0,2*math.pi)/(2*math.pi)
  check('回线与空间积分',f'eddy-space-time-{d}-{w}',pe,d*d*w*w/24)
for A,K in [(.1,3),(1,1),(5,.2),(4,7)]:
 delta=math.sqrt(A/K)
 # 已知壁形代回交换项和各向异性项；截断±20delta，尾部误差小于1e-15。
 sig=quad(lambda x:2*K/math.cosh(x/delta)**2,-20*delta,20*delta)
 check('回线与空间积分',f'wall-energy-{A}-{K}',sig,4*math.sqrt(A*K),1e-6)
 for t,c,ms in [(.1,1,2),(1,.3,1),(3,2,.5)]:
  wall=4*math.sqrt(A*K);ds=math.sqrt(wall*t/(c*ms*ms));energy=lambda d:wall*t/d+c*ms*ms*d
  h=ds*1e-5
  check('有限差分',f'domain-min-derivative-{A}-{K}-{t}',(energy(ds+h)-energy(ds-h))/(2*h),0.,1e-6)
  check('代数与统计积分',f'domain-energy-{A}-{K}-{t}',energy(ds),2*math.sqrt(c*ms*ms*wall*t))
for br,mu,ell,g,ni in [(1,1,2,.3,-.1),(.8,1.5,1,2,-.1),(2,.6,.5,3,.1),(1.2,2,3,.8,.4)]:
 # 独立解磁路/材料二阶矩阵，未知H_m,B。
 mat=np.array([[ell,g],[-mu,1.]])
 h,b=np.linalg.solve(mat,np.array([ni,br]));D=ell+mu*g
 check('磁路矩阵',f'mag-H-{br}-{mu}',h,(ni-br*g)/D)
 check('磁路矩阵',f'mag-B-{br}-{mu}',b,(br*ell+mu*ni)/D)
 e=lambda gap:.5*(br*ell/(ell+mu*gap))**2*gap/ell
 gs=ell/mu;eps=gs*1e-5
 check('有限差分',f'gap-optimum-{br}-{mu}',(e(gs+eps)-e(gs-eps))/(2*eps),0.)
 check('代数与统计积分',f'gap-maxenergy-{br}-{mu}',e(gs),br*br/(8*mu))
 for xk in [.2*br/mu,.7*br/mu]:
  xs=min(xk,br/(2*mu));q=lambda xx:xx*(br-mu*xx)
  sample=np.linspace(0,xk,50001)
  check('极值采样',f'BH-limit-{br}-{mu}-{xk}',max(q(sample)),q(xs),1e-8)
for chi,nd,mr in [(1,.3,2),(.2,.5,1),(5,0,3),(2,1,1)]:
 m,h=np.linalg.solve(np.array([[1,-chi],[nd,1.]]),np.array([mr,0.]))
 check('磁路矩阵',f'remanence-M-{chi}-{nd}',m,mr/(1+nd*chi))
 check('磁路矩阵',f'remanence-H-{chi}-{nd}',h,-nd*mr/(1+nd*chi))
 check('代数与统计积分',f'remanence-B-{chi}-{nd}',m+h,(1-nd)*mr/(1+nd*chi))
for h in [.05,.2,.5,.7,.9]:
 en=lambda theta:math.sin(theta)**2+2*h*math.cos(theta)
 theta=math.acos(h)
 check('代数与统计积分',f'barrier-{h}',en(theta)-en(0),(1-h)**2)
 check('有限差分',f'saddle-derivative-{h}',(en(theta+1e-5)-en(theta-1e-5))/(2e-5),0.)
for th in [.1,.4,.8,1.3,2.1,3.8,5.4]:
 sx,cx=math.sin(th),math.cos(th)
 # 临界场由两个稳定性方程解出，核对参数曲线。
 hx,hz=np.linalg.solve(np.array([[cx,-sx],[sx,cx]]),np.array([sx*cx,-math.cos(2*th)]))
 check('稳定性矩阵',f'astroid-hx-{th}',hx,sx**3)
 check('稳定性矩阵',f'astroid-hz-{th}',hz,-cx**3)
 check('稳定性矩阵',f'astroid-equation-{th}',abs(hx)**(2/3)+abs(hz)**(2/3),1.)
 eps=1e-4;en=lambda t:math.sin(t)**2-2*(hx*math.sin(t)+hz*math.cos(t))
 check('有限差分',f'astroid-first-{th}',(en(th+eps)-en(th-eps))/(2*eps),0.)
 check('有限差分',f'astroid-second-{th}',(en(th+eps)-2*en(th)+en(th-eps))/eps**2,0.,5e-7)
for theta in [.1,.3,.8,1.3,2.5]:
 lam=.2;eps=1e-5;strain=lambda t:1.5*lam*(math.cos(t)**2-1/3)
 check('有限差分',f'magnetostriction-{theta}',(strain(theta+eps)-strain(theta-eps))/(2*eps),-1.5*lam*math.sin(2*theta))
 R=lambda t:.7+1.2/2*(1-math.cos(t))
 check('有限差分',f'MR-sensitivity-{theta}',(R(theta+eps)-R(theta-eps))/(2*eps),.6*math.sin(theta))
# 题给数值独立计算，核对排入文本的舍入值。
mu0=4*math.pi*1e-7;kB=1.380649e-23
vol=math.pi*(10e-9)**3/6;thetaE=5e4*vol/kB
check('题给数值','TB-1s',thetaE/math.log(1e9),91.5,6e-4)
check('题给数值','TB-ms',thetaE/math.log(1e6),137.,3e-3)
check('题给数值','tau-300K',1e-9*math.exp(thetaE/300),5.56e-7,1e-9)
xr=math.log(1e17);xw=math.log(1e3);vmin=kB*300*xr/5e4;vmax=kB*300*xw/(5e4*.3**2)
check('题给数值','write-min-field',1-math.sqrt(xw/xr),.580,3e-4)
check('题给数值','diameter-min-nm',(6*vmin/math.pi)**(1/3)*1e9,18.4,.003)
check('题给数值','diameter-max-nm',(6*vmax/math.pi)**(1/3)*1e9,23.,.003)
B=.72*.01/.012;check('题给数值','gap-field',B,.6)
check('题给数值','gap-force',B*B*1e-4/(2*mu0),14.3,.002)
check('题给数值','reverse-ampere-turn',(.36*.012-.72*.002)/mu0,2290.,.002)
check('题给数值','pickup-integral',-200*1e-4*(.72*.01/.016-B),.003)
groups={}
for r in records:
 g=groups.setdefault(r['group'],{'count':0,'maximum_normalized_error':0.});g['count']+=1;g['maximum_normalized_error']=max(g['maximum_normalized_error'],r['normalized_error'])
result={'count':len(records),'all_passed':True,'count_description':'同一关系在不同参数下的积分、矩阵求解、有限差分和数值比较实例，非独立公式数量','normalized_error_definition':'abs(got-expected)/max(1,abs(got),abs(expected));单位化预期零值使用绝对误差','maximum_normalized_error':max(r['normalized_error'] for r in records),'groups':groups,'records':records}
(ROOT/'检查/第十七讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
