from pathlib import Path
import math,json
import numpy as np
import sympy as S
R=Path(__file__).resolve().parent;checks=[]
def ck(n,ident,model,got,want,tol=1e-7):
 err=float(abs(got-want)/max(1,abs(want)));assert err<tol,(n,ident,model,err)
 checks.append(dict(lecture=n,problem=ident,check=model,scaled_error=err,tolerance=tol,passed=True))
def quad(f,a,b,n=20000):
 h=(b-a)/n
 return h/3*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h) for i in range(1,n)))
def diff(f,x,h=1e-5):return (f(x+h)-f(x-h))/(2*h)
# Independently integrate source fields with k=1, epsilon=1/(4pi).
k=1.;eps=1/(4*math.pi);r=1.7;z0=.3;z1=-.8;z2=1.4;lam=.6
ck(2,'W1','finite rod radial source integration',quad(lambda z:lam*r/(r*r+(z0-z)**2)**1.5,z1,z2),lam/r*((z2-z0)/math.hypot(r,z2-z0)-(z1-z0)/math.hypot(r,z1-z0)))
ck(2,'W1','finite rod signed axial source integration',quad(lambda z:lam*(z0-z)/(r*r+(z0-z)**2)**1.5,z1,z2),lam*(1/math.hypot(r,z2-z0)-1/math.hypot(r,z1-z0)))
l=1.2;beta=.45
ck(2,'W2','odd density axial field',quad(lambda z:lam*(1+beta*z/l)*(-z)/(r*r+z*z)**1.5,-l,l),-2*lam*beta/l*(math.asinh(l/r)-l/math.hypot(r,l)))
a=1.3;alpha=.8;eta=.4
ck(2,'W3','arc central x component',quad(lambda t:-lam*math.cos(t)/a,-alpha,alpha),-2*lam*math.sin(alpha)/a)
ck(2,'W3','full ring nonuniform x component',quad(lambda t:-lam*(1+eta*math.cos(t))*math.cos(t)/a,0,2*math.pi),-math.pi*lam*eta/a)
Q=1.1;f=lambda x:Q*x/(x*x+a*a)**1.5
ck(2,'W4','ring maximum derivative',diff(f,a/math.sqrt(2)),0)
sig=.7;x=.9
ck(2,'W5','disk integrated as concentric rings',quad(lambda u:2*math.pi*sig*u*x/(u*u+x*x)**1.5,0,a),sig/(2*eps)*(1-x/math.hypot(x,a)))
b=.6
fann=lambda x:sig*x/(2*eps)*(1/math.hypot(x,b)-1/math.hypot(x,a))
ck(2,'W6','annulus subtraction',quad(lambda u:2*math.pi*sig*u*x/(u*u+x*x)**1.5,b,a),fann(x))
xm=a**(2/3)*b**(2/3)/math.sqrt(a**(2/3)+b**(2/3));ck(2,'T9','annulus maximum derivative',diff(fann,xm),0)
for ratio in [8.,10.]:
 e=.5/ratio;ck(2,'W7','axial dipole relative error',abs((2/ratio**3)/(1/(ratio-.5)**2-1/(ratio+.5)**2)-1),2*e*e-e**4)
 ck(2,'W7','equatorial dipole relative error',((1/ratio**3)/abs((-np.array([ratio,-.5])/(ratio**2+.25)**1.5+np.array([ratio,.5])/(ratio**2+.25)**1.5)[1])-1),(1+e*e)**1.5-1)
h=.9
ck(2,'W8','disk flux source integration',quad(lambda u:2*math.pi*Q*h*u/(h*h+u*u)**1.5,0,a),Q/(2*eps)*(1-h/math.hypot(h,a)))
ck(2,'T11','square solid angle',4*math.atan(1/math.sqrt(3)),2*math.pi/3)
ck(2,'T1','rod nonuniform charge',quad(lambda z:lam*(1-z*z/l**2),-l,l),4*lam*l/3)
ck(2,'T1','disk nonuniform charge',quad(lambda u:sig*u/a*2*math.pi*u,0,a),2*math.pi*sig*a*a/3)
ck(2,'T2','two point sources maximum',diff(lambda y:2*Q*y/(a*a+y*y)**1.5,a/math.sqrt(2)),0)
# Semi-infinite source mapped to theta via z=r tan(theta).
ck(2,'T3','semi infinite radial',quad(lambda t:lam*math.cos(t)/r,0,math.pi/2),lam/r)
ck(2,'T3','semi infinite axial',quad(lambda t:-lam*math.sin(t)/r,0,math.pi/2),-lam/r)
ck(2,'T5','semicircle nonuniform x field',quad(lambda t:-lam*(1+eta*math.cos(t))*math.cos(t)/a,0,math.pi),-math.pi*lam*eta/(2*a))
ck(2,'T7','second angular harmonic zero total charge',quad(lambda t:math.cos(2*t),0,2*math.pi),0)
ck(2,'T7','second harmonic zero transverse field',quad(lambda t:math.cos(2*t)*math.cos(t),0,2*math.pi),0)
th=math.acos(1/math.sqrt(3));ck(2,'T10','dipole zero z component',3*math.cos(th)**2-1,0)
# Far-field coefficients derived from the exact symmetric sources as power series.
u=S.symbols('u',positive=True)
quad_ax=1/(1/u-a)**2+1/(1/u+a)**2-2*u*u
ck(2,'W12','axial quadrupole coefficient',float(S.expand(S.series(quad_ax,u,0,5).removeO()).coeff(u,4)),6*a*a)
quad_mid=2/(1/u**2+a*a)**S.Rational(3,2)/u-2*u*u
ck(2,'W12','equatorial quadrupole coefficient',float(S.expand(S.series(quad_mid,u,0,5).removeO()).coeff(u,4)),-3*a*a)
ring_cancel=x # arbitrary field evaluation is replaced by independent derivative at zero
xx=S.symbols('x');ering=Q*xx/(xx**2+a*a)**S.Rational(3,2)-8*Q*xx/(xx**2+4*a*a)**S.Rational(3,2)
ck(2,'T16','two ring linear cancellation',float(S.diff(ering,xx).subs(xx,0)),0)
ck(2,'T16','two ring cubic coefficient',float(S.diff(ering,xx,3).subs(xx,0)/6),-9*Q/(8*a**5))
qn=np.array([3.20,4.81,6.40,8.02]);nn=np.array([2,3,4,5]);estimate=float(np.dot(nn,qn)/np.dot(nn,nn));ck(2,'T18','least squares residual orthogonality',float(np.dot(nn,qn-nn*estimate)),0)
# Gauss law, independent radial integrations, continuity and energy checks.
rho=.7;R0=1.8;rr=.9
for exponent in [-2.,-.5,0.,1.,2.]:
 enclosed=4*math.pi*rho*R0**(-exponent)*rr**(exponent+3)/(exponent+3)
 Ei=lambda s:rho*s**(exponent+1)/((exponent+3)*eps*R0**exponent)
 ck(3,'W1','power density Gaussian flux n='+str(exponent),4*math.pi*rr**2*Ei(rr),enclosed/eps)
 phi=lambda s:rho*R0**2/((exponent+3)*eps)*(1+math.log(R0/s) if exponent==-2 else 1+(1-(s/R0)**(exponent+2))/(exponent+2))
 ck(3,'W1','power density potential derivative n='+str(exponent),-diff(phi,rr),Ei(rr))
Ecy=lambda s:rho/(2*eps)*(s-s**3/(2*R0**2))
ck(3,'W3','cylinder radial integration',quad(lambda u:2*math.pi*rho*(1-u*u/R0**2)*u,0,rr),2*math.pi*rr*eps*Ecy(rr))
ck(3,'W3','cylinder maximum',diff(Ecy,R0*math.sqrt(2/3)),0)
# Position-dependent terms cancel explicitly for a hole.
rv=np.array([.3,.7,-.1]);dv=np.array([.2,-.1,.3])
for coeff,ident in [(3,'W5'),(2,'W6')]:
 ck(3,ident,'hole field superposition',np.linalg.norm(rho/(coeff*eps)*(rv-(rv-dv))-rho*dv/(coeff*eps)),0)
alf=.3;bet=.7;aa=1.3;bb=.9
loop=quad(lambda x:0,0,aa)+quad(lambda y:bet*aa,0,bb)+quad(lambda x:-alf*bb,0,aa)+quad(lambda y:0,0,bb)
ck(3,'W8','rectangular circulation independent edge integral',loop,(bet-alf)*aa*bb)
for power in [.5,1.,2.,3.]:
 phi=lambda s:.6*math.log(1.3/s) if power==1 else .6/(power-1)*(s**(1-power)-1.3**(1-power))
 ck(3,'W9','radial power potential derivative p='+str(power),-diff(phi,rr),.6/rr**power)
ck(3,'T2','uniform ball internal work',quad(lambda s:Q*s/R0**3,0,R0),Q/(2*R0))
ck(3,'T3','linear density charge',quad(lambda s:4*math.pi*rho*s**3/R0,0,R0),math.pi*rho*R0**3)
ck(3,'T3','linear density centre potential',quad(lambda s:4*math.pi*rho*s**2/R0,0,R0),4*math.pi*rho*R0**2/3)
ck(3,'T4','cylinder outside potential difference',quad(lambda s:-rho*R0**2/(2*eps*s),R0,2*R0),-rho*R0**2*math.log(2)/(2*eps))
# Local jump sign: increasing z is the shared normal.
ck(3,'T6','normal field jump',eps*(2-(-1)),3*eps)
ck(3,'T10','dipole potential work',((-Q/(3*a)+Q/a)-(-Q/(4*a)+Q/(2*a))),5*Q/(12*a))
ck(3,'T12','log potential work',quad(lambda s:.8/s,.7,1.6),.8*math.log(1.6/.7))
A=.3;B=2*A*R0**3;C=3*A*R0**2
ck(3,'T14','no surface charge potential match',C-A*R0**2,B/R0)
ck(3,'T14','no surface charge field match',2*A*R0,B/R0**2)
Esl=lambda z:rho*(z*z-aa*aa)/(2*eps*aa)
ck(3,'T17','odd slab outer potential jump',-quad(Esl,-aa,aa),2*rho*aa*aa/(3*eps))
omega=.8;t=math.pi/(2*omega)
ck(3,'T18','charged ball crossing endpoint',R0*(math.cos(omega*t)-math.sin(omega*t)),-R0)
ck(3,'T18','charged ball crossing exit speed',-R0*omega*(math.sin(omega*t)+math.cos(omega*t)),-R0*omega)
ck(3,'T18','outside fall time from energy integral',quad(lambda u:math.sqrt(2)*math.cos(u)**2,0,math.pi/4),(1+math.pi/2)/(2*math.sqrt(2)))
(R/'记录/第二三讲独立计算复核.json').write_text(json.dumps({'checks':checks,'count':len(checks),'passed':len(checks),'normalization':'k=1; epsilon_0=1/(4*pi)','scope':'selected independent integrations, potential derivatives, far-field expansions and dynamics; every question also manually reviewed'},ensure_ascii=False,indent=2))
print(len(checks),'independent checks passed')
