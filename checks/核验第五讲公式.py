"""独立积分、有限差分和节点求解检查；使用一致的归一化单位。"""
from pathlib import Path
import math,json

ROOT=Path(__file__).resolve().parent
k=1.;eps=1/(4*math.pi);records=[]
def check(label,actual,expected,tol=2e-7):
    error=abs(actual-expected)/max(1,abs(expected))
    records.append(dict(label=label,actual=actual,expected=expected,
                        scaled_error=error,tolerance=tol,passed=error<tol))
    assert error<tol,(label,actual,expected,error)
def integ(f,lo,hi,n=10000):
    h=(hi-lo)/n
    return h/3*(f(lo)+f(hi)+sum((4 if i%2 else 2)*f(lo+i*h) for i in range(1,n)))
def diff(f,x,h=1e-5):
    return (f(x+h)-f(x-h))/(2*h)

# 从完整镜像势求导，与简化的壁密度和力比较。
a=1.3
for q in [.8,-1.1]:
    for s in [.15,.6,1.1]:
        qp=-q*a/s;sp=a*a/s
        def phi(r,theta):
            return k*(q/math.sqrt(r*r+s*s-2*r*s*math.cos(theta))
                     +qp/math.sqrt(r*r+sp*sp-2*r*sp*math.cos(theta)))
        for theta in [0.,.7,1.8,math.pi]:
            check('W4 球壁势抵消',phi(a,theta),0)
            R=math.sqrt(a*a+s*s-2*a*s*math.cos(theta))
            sig=-q*(a*a-s*s)/(4*math.pi*a*R**3)
            check('W4 内壁密度与势导数',eps*diff(lambda rr:phi(rr,theta),a),sig)
        sig=lambda u:-q*(a*a-s*s)/(4*math.pi*a*(a*a+s*s-2*a*s*u)**1.5)
        check('W4 内壁总量积分',2*math.pi*a*a*integ(sig,-1,1),-q)
        image_potential=lambda z:k*qp/(sp-z)
        check('W4 去自场后的镜像力',-q*diff(image_potential,s),
              k*q*q*a*s/(a*a-s*s)**2)
for s in [.01,.4,.65]:
    check('T7 近远密度比',((a*a+s*s+2*a*s)/(a*a+s*s-2*a*s))**1.5,
          ((a+s)/(a-s))**3)

# 厚壳：从场积分复核共同势与腔内分段势。
q=.9;Qs=-.3;b=2.;r=.7
Vc=k*(q+Qs)/b
check('W2 金属区势连续',Vc+k*q*(1/a-1/a),Vc)
check('W2 腔内积分势',Vc+integ(lambda rr:k*q/rr**2,r,a),
      Vc+k*q*(1/r-1/a))
check('W2 接地外壁总量',(-q)+q,0)

# 连通两球：直接解等势和守恒方程。
ra=.7;rb=1.2;D=10.;Q=1.4
qa=Q*(1/rb-1/D)/(1/ra+1/rb-2/D);qb=Q-qa
H=D*(ra+rb)-2*ra*rb
check('W5 等势方程解Qa',qa,Q*ra*(D-rb)/H)
check('W5 等势方程解Qb',qb,Q*rb*(D-ra)/H)
check('W5 两球电势一致',qa/ra+qb/D,qb/rb+qa/D)

# 各电容通过原始高斯场积分，避免仅重算同一比值。
aa=.8;bb=1.9;L=4.3;charge=.7
Vs=integ(lambda rr:k*charge/rr**2,aa,bb)
check('球形电容场积分',charge/Vs,4*math.pi*eps*aa*bb/(bb-aa))
lam=charge/L
Vco=integ(lambda rr:lam/(2*math.pi*eps*rr),aa,bb)
check('同轴电容场积分',charge/Vco,2*math.pi*eps*L/math.log(bb/aa))
bstar=3.;astar=bstar/math.e
check('W6 最优耐压一阶导数',diff(lambda rr:rr*math.log(bstar/rr),astar),0)
check('W6 最优耐压数值',astar*math.log(bstar/astar),bstar/math.e)
check('T11 近似与精确电容比',
      (2*math.pi*eps)/(2*math.pi*eps/math.log(2)),math.log(2))

# 三个薄球壳：从带电球壳势叠加反算各表面势。
aa=.6;bb=1.1;cc=1.9
C1=4*math.pi*eps*aa*bb/(bb-aa)
C2=4*math.pi*eps*bb*cc/(cc-bb);C3=4*math.pi*eps*cc
Va=1.2;Vb=-.3;Vc=.4
Qa=C1*(Va-Vb);Qb=C1*(Vb-Va)+C2*(Vb-Vc);Qc=C2*(Vc-Vb)+C3*Vc
check('W7 内球势叠加',k*(Qa/aa+Qb/bb+Qc/cc),Va)
check('W7 中球势叠加',k*((Qa+Qb)/bb+Qc/cc),Vb)
check('W7 外球势叠加',k*(Qa+Qb+Qc)/cc,Vc)
for middle in [.8,1.2,1.6]:
    u=aa*middle/(middle-aa);v=middle*cc/(cc-middle)
    check('W7 中性薄壳不改变两端电容',u*v/(u+v),aa*cc/(cc-aa))
# 厚壳去掉实际金属区的积分。
aa=.5;bb=.9;cc=1.2;dd=1.8
V=integ(lambda rr:k*charge/rr**2,aa,bb)+integ(lambda rr:k*charge/rr**2,cc,dd)
check('T12 厚壳电容分段积分',charge/V,4*math.pi*eps/(1/aa-1/bb+1/cc-1/dd))

# 电容桥：两中点联立求解，不假设先验中点势。
def bridge(c1,c2,c3,c4,cg,U):
    A=c1+c2+cg;B=c3+c4+cg;det=A*B-cg*cg
    vm=U*(c1*B+cg*c3)/det
    vn=U*(c3*A+cg*c1)/det
    return vm,vn,c1*(U-vm)+c3*(U-vn)
vm,vn,Qa=bridge(1,2,2,1,1,1)
check('W9 桥中点M',vm,2/5);check('W9 桥中点N',vn,3/5)
check('W9 桥电量',vm-vn,-1/5);check('W9 端口电容',Qa,7/5)
for cg in [.3,1.,5.]:
    vm,vn,Qa=bridge(2,1,4,2,cg,1)
    check('T14 平衡桥电压',vm-vn,0)
    check('T14 平衡桥端口电容',Qa,2)

# 带电浮置节点与片：从电容节点先求势，再求电场和压力。
S=1.4;d1=.7;d2=1.2;D=d1+d2
for q0,U in [(0.,.8),(.7,0.),(-.4,.8),(.9,-.3)]:
    c1=eps*S/d1;c2=eps*S/d2
    vm=(q0+c1*U)/(c1+c2);e1=(U-vm)/d1;e2=vm/d2
    check('W12 中间片势',vm,U*d2/D+q0*d1*d2/(eps*S*D))
    check('W12 片净电量',eps*S*(e2-e1),q0)
    check('W12 电压路径积分',e1*d1+e2*d2,U)
    F=eps*S*(e2*e2-e1*e1)/2
    check('W12 压力与去自场力',F,q0*(e1+e2)/2)
    check('W12 力的化简',F,q0*U/D+q0*q0*(d1-d2)/(2*eps*S*D))
q0=.7;D=1.6
for x in [-.3,0,.2]:
    d1=D/2+x;d2=D/2-x;c1=eps*S/d1;c2=eps*S/d2
    vm=q0/(c1+c2);e1=-vm/d1;e2=vm/d2
    check('T17 位移后压力差',eps*S*(e2*e2-e1*e1)/2,q0*q0*x/(eps*S*D))
for c1,c2,q0,U in [(1.,2.,.4,.7),(1.,1.,3.,1.)]:
    vm=(q0+c1*U)/(c1+c2);q1=c1*(U-vm);q2=c2*vm
    check('W10 中点电量守恒',q2-q1,q0)
    qin=lambda uu:c1*(uu-(q0+c1*uu)/(c1+c2))
    check('W10 固定预电量增量电容',diff(qin,U),c1*c2/(c1+c2))
for polarity in [1,-1]:
    c1=1.;c2=2.;u1=2.;u2=1.
    qA=c1*u1+polarity*c2*u2;uf=qA/(c1+c2)
    check('T16 合并节点电量守恒',(c1+c2)*uf,qA)
    check('T16 同向反向终态',uf,4/3 if polarity==1 else 0)

# 匀强外场中的带电球：势的数值导数和密度总量。
Q=.9;E0=.7;a=1.1
phi=lambda r,t:k*Q/r-E0*(r-a**3/r**2)*math.cos(t)
for t in [0.,.9,math.pi]:
    check('T2 球面势一致',phi(a,t),k*Q/a)
    check('T2 表面密度与势导数',-eps*diff(lambda rr:phi(rr,t),a),
          Q/(4*math.pi*a*a)+3*eps*E0*math.cos(t))
sig=lambda t:Q/(4*math.pi*a*a)+3*eps*E0*math.cos(t)
check('T2 表面总量积分',integ(lambda t:sig(t)*2*math.pi*a*a*math.sin(t),0,math.pi),Q)

result=dict(normalization='k=1, epsilon_0=1/(4*pi); consistent normalized units',
            methods=['Simpson integration','central finite differences','independent node solution'],
            count=len(records),passed=sum(x['passed'] for x in records),checks=records)
(ROOT/'检查/第五讲_公式核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'count':result['count'],'passed':result['passed'],
                 'max_scaled_error':max(x['scaled_error'] for x in records)},ensure_ascii=False))
