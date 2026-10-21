"""独立数值核验：用角积分、空间积分和有限差分检查讲义公式。"""
from pathlib import Path
import math,json
checks=[]
def check(name,got,want,tol=1e-7):
    err=abs(got-want)/max(1,abs(got),abs(want))
    checks.append(dict(name=name,computed=got,expected=want,error=err,passed=err<=tol))
def integral(f,a,b,n=12000):
    h=(b-a)/n
    return h/3*(f(a)+f(b)+4*sum(f(a+(2*i-1)*h) for i in range(1,n//2+1))+2*sum(f(a+2*i*h) for i in range(1,n//2)))
def der(f,x,h=1e-5):return (f(x+h)-f(x-h))/(2*h)
def m(R,r,eta):
    if r==0:return R**(-1-eta)/(1+eta)
    return ((R+r)**(1-eta)-abs(R-r)**(1-eta))/(2*R*r*(1-eta*eta))
def S(x):return ((1+x)*math.log1p(x)-(1-x)*math.log1p(-x))/(2*x)-math.log(2)
for eta in [-.12,-.01,0,.03,.15]:
    for r in [.15,.6,1.4,2.5]:
        # 直接角积分，与距离积分解析式独立。
        val=integral(lambda u:(1+r*r-2*r*u)**(-(1+eta)/2)/(2*(1+eta)),-1,1)
        check(f'shell angular eta={eta},r={r}',val,m(1,r,eta))
for x in [.05,.2,.5,.8,.95]:
    def response(eta):return 1-m(1,x,eta)/m(1,1,eta)
    check(f'null response derivative x={x}',der(response,0,2e-5),S(x),1e-8)
    eta=.002
    A=m(1,1,eta);B=m(x,x,eta);C=m(1,x,eta);det=A*B-C*C
    qa=(B-C)/det;qb=(A-C)/det
    check(f'initial outer potential x={x}',A*qa+C*qb,1)
    check(f'initial inner potential x={x}',C*qa+B*qb,1)
    qap=-C*qb/A
    check(f'grounded outer x={x}',A*qap+C*qb,0)
    check(f'inner readout x={x}',C*qap+B*qb,response(eta))
    check(f'charge fraction x={x}',qb/(qa+qb),(A-C)/(A+B-2*C))
for eta in [-.08,.08]:
    # 小r场首项和角积分势的有限差分。
    r=.001
    E=-der(lambda z:integral(lambda u:(1+z*z-2*z*u)**(-(1+eta)/2)/(2*(1+eta)),-1,1,4000),r,2e-4)
    check(f'inner field linear eta={eta}',E,-eta*r/3,1e-8)
for x in [.0005,.001,.002]:
    check(f'S small x={x}',S(x),1-math.log(2)-x*x/6,1e-10)
for h in [.00001,.00003,.0001]:
    check(f'S narrow gap h={h}',S(1-h),h/2*(math.log(2/h)-1),1e-7)
check('W2 exponent bound',2e-7/S(.8),1.1813868e-6,1e-11)
check('T3 bound',5e-7/S(.5),1.9111404e-6,1e-11)
check('T18 bound',2.5e-7/S(.5),9.555702e-7,1e-11)
for a,b,L in [(1,2,3),(2,1,4),(.7,.7,2),(.4,3,1)]:
    rr=lambda x:a+(b-a)*x/L
    R=integral(lambda x:1/(math.pi*rr(x)**2),0,L)
    check(f'taper R {a,b,L}',R,L/(math.pi*a*b))
    phi=lambda x:a*(L-x)/(L*rr(x))
    for x in [L*.15,L*.45,L*.85]:
        check(f'taper gradient {a,b,L,x}',-der(phi,x),a*b/(L*rr(x)**2))
    # 固定单位端压的全体发热，与I*U比较。
    I=math.pi*a*b/L
    P=integral(lambda x:I*I/(math.pi*rr(x)**2),0,L)
    check(f'taper Joule {a,b,L}',P,I)
for a,b,c,g1,g2 in [(1,3,2,1,2),(.6,4,1,2,.4),(2,6,4,.7,.7)]:
    R=integral(lambda r:1/(4*math.pi*g1*r*r),a,c)+integral(lambda r:1/(4*math.pi*g2*r*r),c,b)
    formula=((1/a-1/c)/g1+(1/c-1/b)/g2)/(4*math.pi)
    check(f'sphere layers R {a,b,c}',R,formula)
    I=1/formula
    P=integral(lambda r:I*I/(4*math.pi*g1*r*r),a,c)+integral(lambda r:I*I/(4*math.pi*g2*r*r),c,b)
    check(f'sphere layers power {a,b,c}',P,I)
for a,b,L in [(.5,1,2),(1,3,5),(2,2.001,1)]:
    R=integral(lambda r:1/(2*math.pi*r*L),a,b)
    check(f'cylinder R {a,b,L}',R,math.log(b/a)/(2*math.pi*L))
    U=3;I=U/R
    P=integral(lambda r:U*U/(r*r*math.log(b/a)**2)*2*math.pi*r*L,a,b)
    check(f'cylinder Joule {a,b,L}',P,U*I)
check('graded rod integral',integral(lambda u:(1+u)*math.exp(-u),0,1),2-3/math.e)
for u in [.1,.4,.8]:
    phi=lambda z:1-(2-(z+2)*math.exp(-z))/(2-3/math.e)
    check(f'graded rod gradient {u}',-der(phi,u),(1+u)*math.exp(-u)/(2-3/math.e))
check('W9 angle',math.atan(2)*180/math.pi,63.4349488229)
check('W9 field ratio',math.hypot(.25,.5)/math.hypot(1,.5),.5)
for x,y in [(1,0),(1,2),(-2,3)]:
    jx=2*x+y;jy=x+2*y
    check(f'tensor heat {x,y}',jx*x+jy*y,((x+y)**2*3+(x-y)**2)/2)
# 圆孔的解析势，用笛卡尔有限差分独立检验Laplace方程。
def phi(x,y):return -x*(1+1/(x*x+y*y))
for x,y in [(1.4,.3),(.3,1.8),(-2,.7)]:
    h=2e-4
    lap=(phi(x+h,y)+phi(x-h,y)+phi(x,y+h)+phi(x,y-h)-4*phi(x,y))/h**2
    check(f'hole Laplace {x,y}',lap,0,3e-7)
for theta in [0,.3,.8,1.57,2.3]:
    f=lambda r:-(r+1/r)*math.cos(theta)
    check(f'hole zero normal theta={theta}',-der(f,1),0)
    check(f'hole tangent theta={theta}',-der(lambda t:-2*math.cos(t),theta),-2*math.sin(theta))
for I in [.2,.5,.8]:
    R0=2;alpha=.02;H=1
    T=I*I*R0/(H-alpha*I*I*R0)
    check(f'thermal I balance {I}',I*I*R0*(1+alpha*T),H*T)
for U in [.5,2,6]:
    R0=2;alpha=.02;H=1
    T=(math.sqrt(1+4*alpha*U*U/(H*R0))-1)/(2*alpha)
    check(f'thermal U balance {U}',U*U/(R0*(1+alpha*T)),H*T)
check('nonlinear load point',2*1+1**3,3)
def nonlinear_voltage(emf):
    u=1.0
    for _ in range(12):u-=(u**3+2*u-emf)/(3*u*u+2)
    return u
check('nonlinear small response',der(nonlinear_voltage,3),.2)
check('two battery energy',12,9+1+1+1)
R1=1/(8*math.pi);R2=1/(48*math.pi)
check('timed layers ratio',R1/R2,6)
check('timed layers sum',R1+R2,7/(48*math.pi))
I=2;r0=.2;R=R1+R2;emf=I*(r0+R)
check('timed source heat',emf*I,I*I*(r0+R))
# 奇偶读出与最坏误差界，不假设独立随机误差。
for eplus,eminus in [(-.01,.01),(.01,-.01),(.005,.005)]:
    eta=.03;v0=.2;V=10;s=S(.5)
    vp=eta*s*V+v0+eplus;vm=-eta*s*V+v0+eminus
    est=(vp-vm)/(2*s*V)
    bound=.01/(s*V)
    check(f'polarity estimate {eplus,eminus}',est-eta,(eplus-eminus)/(2*s*V))
    checks.append(dict(name=f'polarity bound {eplus,eminus}',computed=abs(est-eta),expected=bound,error=0,passed=abs(est-eta)<=bound+1e-14))
report=dict(count=len(checks),all_passed=all(c['passed'] for c in checks),maximum_normalized_error=max(c['error'] for c in checks),checks=checks)
path=Path(__file__).resolve().parent/'检查/第七讲_公式核验.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:report[k] for k in ['count','all_passed','maximum_normalized_error']},ensure_ascii=False))
for c in checks:
    if not c['passed']:print(c)
assert report['all_passed']

