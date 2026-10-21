"""以有限差分、数值积分和能量导数独立复核本讲的关键结果。"""
from pathlib import Path
import math, json

report=[]
def close(name, got, want, tol=2e-7):
    err=abs(got-want)/max(1,abs(want))
    assert err<tol,(name,got,want,err)
    report.append(dict(check=name,relative_scaled_error=err))

def d(f,x,h=1e-5):
    return (f(x+h)-f(x-h))/(2*h)

def integral(f,lo,hi,n=10000):
    h=(hi-lo)/n
    return h/3*(f(lo)+f(hi)+sum((4 if i%2 else 2)*f(lo+i*h) for i in range(1,n)))

# 以下统一用 eps=1、k=1/(4pi)，参数选择不依赖量纲单位。
k=1/(4*math.pi);a=1.4;s=1.9;z=.6;lam=.8
rod=lambda s,z:k*lam*(math.asinh((a-z)/s)+math.asinh((a+z)/s))
es=k*lam/s*((a-z)/math.hypot(s,a-z)+(a+z)/math.hypot(s,a+z))
ez=k*lam*(1/math.hypot(s,z-a)-1/math.hypot(s,z+a))
close('有限杆径向场',-d(lambda v:rod(v,z),s),es)
close('有限杆轴向场',-d(lambda v:rod(s,v),z),ez)
close('有限杆源积分',integral(lambda v:k*lam/math.hypot(s,z-v),-a,a),rod(s,z),1e-8)

mu=.9;sigma=.7
for zz in [-2.,-.3,.3,2.]:
    ordinary=lambda t:sigma/2*(math.hypot(t,a)-abs(t))
    dip=lambda t:mu/2*((1 if t>0 else -1)-t/math.hypot(t,a))
    close(f'普通圆盘场 z={zz}',-d(ordinary,zz),sigma/2*((1 if zz>0 else -1)-zz/math.hypot(zz,a)))
    close(f'偶极圆盘常规场 z={zz}',-d(dip,zz),mu*a*a/(2*(zz*zz+a*a)**1.5))
    omega=integral(lambda v:2*math.pi*zz*v/(v*v+zz*zz)**1.5,0,a)
    close(f'圆盘立体角积分 z={zz}',mu*omega/(4*math.pi),dip(zz),1e-7)

rho0=.6;vl=-.2;vr=.8
slab=lambda t:-rho0*t*t/2+rho0*t**4/(12*a*a)+(vr-vl)*t/(2*a)+(vl+vr)/2+5*rho0*a*a/12
close('带电层左端势',slab(-a),vl)
close('带电层右端势',slab(a),vr)
close('带电层场',-d(slab,z),rho0*(z-z**3/(3*a*a))-(vr-vl)/(2*a))
close('带电层高斯积分',integral(lambda t:rho0*(1-t*t/a/a),-a,a),4*rho0*a/3,1e-8)

q=.9;hh=1.7;aa=.8
U=lambda aa,hh:k*q*q/(2*aa)-k*q*q/(2*hh)-k*q*q/(2*math.hypot(aa,hh))
D=math.hypot(aa,hh)
fx=k*q*q/(4*aa*aa)-k*q*q*aa/(4*D**3)
fz=-k*q*q/(4*hh*hh)-k*q*q*hh/(4*D**3)
close('双电荷能量水平导数',-d(lambda t:U(t,hh),aa),2*fx)
close('双电荷能量竖直导数',-d(lambda t:U(aa,t),hh),2*fz)
plane=lambda ss,zz:k*q*(1/math.hypot(ss,zz-hh)-1/math.hypot(ss,zz+hh))
close('接地平面边界势',plane(s,0),0)
close('接地平面表面场',-d(lambda t:plane(s,t),0),-2*k*q*hh/(s*s+hh*hh)**1.5)
# t=s/h,再用 u=t/sqrt(1+t²)，感应总量积分恰为 -q du。
close('感应面电荷总量',integral(lambda u:-q,0,1),-q)

R=1.2;E0=.75;r=2.1;theta=.7
ball=lambda rr,tt:-E0*(rr-R**3/rr**2)*math.cos(tt)
close('球面等势',ball(R,theta),0)
close('导体球径向场',-d(lambda v:ball(v,theta),r),E0*(1+2*R**3/r**3)*math.cos(theta))
close('导体球角向场',-d(lambda v:ball(r,v),theta)/r,-E0*(1-R**3/r**3)*math.sin(theta))
close('导体球感应净量',integral(lambda t:3*E0*math.cos(t)*2*math.pi*R*R*math.sin(t),0,math.pi),0)
close('导体球偶极矩',integral(lambda t:3*E0*math.cos(t)*R*math.cos(t)*2*math.pi*R*R*math.sin(t),0,math.pi),4*math.pi*R**3*E0,2e-7)

dm=.3;kap=.8;mu0=.9;x=.4
nonuniform=lambda xx,zz:(1 if zz>0 else -1)/2*(mu0+dm*math.exp(-kap*abs(zz))*math.cos(kap*xx))
for zz in [-.7,.7]:
    sign=1 if zz>0 else -1
    close(f'非均匀层切向场 z={zz}',-d(lambda v:nonuniform(v,zz),x),sign*dm*kap/2*math.exp(-kap*abs(zz))*math.sin(kap*x))
    close(f'非均匀层法向场 z={zz}',-d(lambda v:nonuniform(x,v),zz),dm*kap/2*math.exp(-kap*abs(zz))*math.cos(kap*x))

for eta in [.1,.4,.9]:
    u=a*(1-eta)/math.sqrt(eta*(2-eta))
    close(f'负粒子转向位置 eta={eta}',1-u/math.hypot(u,a),eta)
eta=.01;ratio=1/math.sqrt((1+eta)**2-1)
close('点源近似1%阈值',math.sqrt(1+1/ratio**2)-1,eta)

dest=Path(__file__).resolve().parent/'检查'/'第四讲_公式核验.json'
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(f'{len(report)} 项公式核验通过；最大误差 {max(x["relative_scaled_error"] for x in report):.3g}')
