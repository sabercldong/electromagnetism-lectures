from pathlib import Path
import numpy as np,math,json
mu=4*math.pi*1e-7
phi0=6.62607015e-34/(2*1.602176634e-19)
records=[]
def check(group,name,a,b,tol=1e-7):
    err=abs(a-b)/max(1,abs(a),abs(b))
    records.append(dict(group=group,name=name,got=float(a),expected=float(b),normalized_error=float(err),tolerance=tol))
    assert err<tol,(name,a,b,err)
x,w=np.polynomial.legendre.leggauss(100)
def integral(f,a,b):
    pts=(b-a)*x/2+(a+b)/2
    return (b-a)/2*np.dot(w,f(pts))
def derivative(f,z,h=1e-4):
    return (f(z-2*h)-8*f(z-h)+8*f(z+h)-f(z+2*h))/(12*h)
for H in [.2,1,3]:
    val=integral(lambda t:-1.5*H*np.sin(t)**3,0,math.pi)
    check('空间积分','sphere-cancel-'+str(H),val/2,-H)
    check('空间积分','sphere-moment-'+str(H),math.pi*val,-2*math.pi*H)
for d in [.1,.5,1,3,7]:
    B=lambda t:np.cosh(t)/math.cosh(d)
    J=lambda t:-np.sinh(t)/math.cosh(d)
    check('空间积分','slab-average-'+str(d),integral(B,-d,d)/(2*d),math.tanh(d)/d)
    check('能量积分','slab-energy-'+str(d),integral(lambda t:(B(t)**2+J(t)**2)/2,-d,d),math.tanh(d))
    jt=lambda t:np.cosh(t)/(2*math.sinh(d))
    check('空间积分','transport-normalization-'+str(d),integral(jt,-d,d),1)
    check('边界关系','slab-surface-'+str(d),float(B(d)),1)
for t in [.1,.3,.5,.8,.95]:
    f=lambda y:.5*(1-y*y)**2
    check('有限差分','condensation-entropy-'+str(t),-derivative(f,t),2*t*(1-t*t))
for q in [.1,.3,.7,1.2]:
    for z in [.2,.7,1.5]:
        U=lambda s:q*q/(2*(1+s*s)**3)
        check('有限差分','loop-force-'+str((q,z)),-derivative(U,z),3*q*q*z/(1+z*z)**4)
for theta in [0,.4,1,math.pi/2]:
    for z in [.8,1.5,3]:
        U=lambda y:(1+math.cos(theta)**2)/(64*math.pi*y**3)
        check('有限差分','image-force-'+str((theta,z)),-derivative(U,z),3*(1+math.cos(theta)**2)/(64*math.pi*z**4))
for f in [0,.13,.25,.5,.73]:
    ds=np.linspace(0,2*math.pi,200000)
    current=10*np.sin(ds+math.pi*f)+6*np.sin(ds-math.pi*f)
    expect=math.sqrt(136+120*math.cos(2*math.pi*f))
    check('相位极值','unequal-SQUID-'+str(f),float(np.max(current)),expect,1e-8)
for p in [0,.2,.8,1,1.5,2,3]:
    val=integral(lambda z:np.cos(2*math.pi*p*z),-.5,.5)
    expected=1 if p==0 else math.sin(math.pi*p)/(math.pi*p)
    check('空间积分','Fraunhofer-'+str(p),val,expected)
for i in [0,.2,.5,.8,.99]:
    stable=math.asin(i);unstable=math.pi-stable
    U=lambda d:-math.cos(d)-i*d
    barrier=U(unstable)-U(stable)
    check('代数与势能','biased-barrier-'+str(i),barrier,2*(math.sqrt(1-i*i)-i*math.acos(i)))
for M0 in [0,.3,.8,1.5]:
    L1,L2,P,d,g=2.,3.,5.,1.7,.3;A=L1*L2;b=M0*M0
    C=L1*L2*L2*P*P/(g*d)
    y=2*A*A/(2*A*b+C+math.sqrt(C*C+4*A*b*C));z0=-d/2*math.log(y)
    def energy(z):
        t=math.exp(-z/d);matrix=np.array([[L1,M0*t],[M0*t,L2]])
        flux=np.array([P*t,0]);I=np.linalg.solve(matrix,-flux)
        return .5*I@matrix@I
    for z in [.2,1.,z0]:
        t=math.exp(-z/d);D=A-b*t*t
        F=L1*L2*L2*P*P*t*t/(d*D*D)
        check('有限差分','coupled-force-'+str((M0,z)),-derivative(energy,z),F)
    check('平衡关系','coupled-equilibrium-'+str(M0),-derivative(energy,z0),g)
    h=.0005;curvature=(energy(z0+h)-2*energy(z0)+energy(z0-h))/(h*h)
    check('有限差分','coupled-frequency-'+str(M0),curvature,2*g/d*(A+b*y)/(A-b*y),2e-6)
lo,hi=1.,3.
for _ in range(80):
    mid=(lo+hi)/2
    if mid-2*math.sin(mid)>0:hi=mid
    else:lo=mid
root=(lo+hi)/2
check('代数与势能','rf-SQUID-root',root,1.895494267033981)
for delta in [math.pi-root,math.pi+root]:
    check('平衡关系','rf-SQUID-balance-'+str(delta),delta+2*math.sin(delta),math.pi)
    assert 1+2*math.cos(delta)>0
numbers=[
('ring-current-uA',-.3*phi0/1e-9*1e6,-.62035),
('ring-energy',.09*phi0**2/(2e-9)/1e-22,1.9242),
('isotope-alpha',-math.log(4.158/4.2)/math.log(203/199),.5050),
('isotope-predicted',4.2*math.sqrt(199/203),4.1584),
('vortex-spacing-nm',math.sqrt(2*phi0/(math.sqrt(3)*.2))*1e9,109.26),
('array-voltage',2000*phi0*70e9,.2894967),
('rf-SQUID-low-flux',(math.pi-root)/(2*math.pi),.198323),
('rf-SQUID-high-flux',(math.pi+root)/(2*math.pi),.801677)]
for name,a,b in numbers:check('数值舍入',name,a,b,7e-4)
groups={}
for r in records:
    g=groups.setdefault(r['group'],dict(count=0,maximum_normalized_error=0))
    g['count']+=1;g['maximum_normalized_error']=max(g['maximum_normalized_error'],r['normalized_error'])
out=dict(count=len(records),all_passed=True,count_description='不同参数的空间积分、有限差分、相位极值、边界和平衡比较实例，非独立公式数量',groups=groups,records=records)
(Path(__file__).parent/'检查/第十九讲_公式核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='records'},ensure_ascii=False))
