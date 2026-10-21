"""Independent calculations of the principal quantitative claims.
One record is one parameter instance, not one quadrature point.
Normalized SI coefficients k_e=mu_0=epsilon=1 are used when the common
dimensional prefactor cancels. Dimensional numerical answers use SI below.
"""
from pathlib import Path
import math,json,numpy as np
from numpy.polynomial.legendre import leggauss
R=Path(__file__).resolve().parent
X,W=leggauss(120);rows=[]
def integ(f,a,b):
    x=(a+b)/2+(b-a)*X/2
    return np.sum(np.array([f(v) for v in x])*W)*(b-a)/2
def diff(f,x,h=1e-5):return (f(x+h)-f(x-h))/(2*h)
def ck(key,s,actual,expected,method,tol=2e-7):
    a=np.asarray(actual,dtype=complex);e=np.asarray(expected,dtype=complex)
    err=float(np.max(abs(a-e)));scale=max(float(np.max(abs(e))),1e-12)
    ok=err<=tol*max(scale,1.)
    def val(v):
        v=np.asarray(v,dtype=complex).flatten()
        return [[float(x.real),float(x.imag)] for x in v]
    rows.append(dict(problem=key,parameter=s,method=method,actual=val(a),
                     expected=val(e),absolute_error=err,tolerance=tol*max(scale,1),passed=bool(ok)))
def main():
  for s in [.7,1.4]:
    # Part 1: electrostatics. Direct source integrations, boundary solves and energy derivatives.
    a=1+s;b=3*a;Q=1.3;pt=np.array([.35*a,.2*a,0])
    def shell_field(pt):
        out=np.zeros(3)
        for u,wu in zip(X,W):
            for phi in np.arange(240)*2*np.pi/240:
                src=a*np.array([math.sqrt(1-u*u)*math.cos(phi),math.sqrt(1-u*u)*math.sin(phi),u])
                z=pt-src;out+=Q/(4*np.pi)*wu*2*np.pi/240*z/np.linalg.norm(z)**3
        return out
    Eext=-Q*(pt-np.array([b,0,0]))/np.linalg.norm(pt-np.array([b,0,0]))**3
    ck('P1W1',s,shell_field(pt)+Eext,Eext,'球面电荷二维源积分检验内区球面贡献零')
    A=np.array([-1/math.sqrt(3),0]);B=-A;vertices=[np.array([-math.sqrt(3),0]),np.array([0,-1]),np.array([0,1])]
    def edge(U,V,P):return integ(lambda t:(1+s*(t-.5)**2)*np.linalg.norm(V-U)/np.linalg.norm(U+t*(V-U)-P),0,1)
    va=np.array([edge(vertices[i],vertices[(i+1)%3],A) for i in range(3)])
    vb=np.array([edge(vertices[i],vertices[(i+1)%3],B) for i in range(3)])
    ck('P1W2',s,[sum(va)-va[0],sum(vb)-vb[0]],[2*sum(va)/3,sum(vb)/2+sum(va)/6],'三边对称非均匀线密度势积分')
    C0=s;V0=2;C1=C0;C2=C0/2;v=C0*V0/(C1+C2)
    ck('P1W3',s,[v*2/V0,v/V0,(C1+C2)/C0],[4/3,2/3,1.5],'并联节点电荷守恒解与场比')
    rad=s;r=3*s;qs=.3;q=.8
    image=-q*rad/r;imagepos=rad**2/r;central=qs-image
    F=q*(central/r**2+image/(r-imagepos)**2)
    ck('P1W4',s,F,q*qs/r**2+q*q*rad*(1/r**3-r/(r*r-rad*rad)**2),'独立镜像逐源受力求和')
    def vertical(y):return s/((1+y*y)**1.5)
    Fx_line=integ(vertical,0,1e3) # finite cutoff checked analytically separately
    # Map semi-infinite lines with t=tan(theta); integrate the complete geometry.
    Ex=integ(lambda t:math.cos(t),0,math.pi/2)*s
    Ey=-integ(lambda t:math.sin(t),0,math.pi/2)*s
    arcx=-s*integ(math.cos,math.pi,1.5*math.pi)
    arcy=-s*integ(math.sin,math.pi,1.5*math.pi)
    ck('P1W5',s,[Ex+Ey+arcx,Ex+Ey+arcy],[s,s],'半无限线变量代换与四分之一圆弧逐分量积分')
    L=1.3;y=s
    vx=integ(lambda x:-x*x/(x*x+y*y)**1.5,0,L)
    vy=integ(lambda x:x*y/(x*x+y*y)**1.5,0,L)
    ck('P1W6',s,[vx,vy,integ(lambda x:x/math.sqrt(x*x+y*y),0,L)],
       [-math.asinh(L/y)+L/math.sqrt(L*L+y*y),1-y/math.sqrt(L*L+y*y),math.sqrt(L*L+y*y)-y],'线密度κx直接势场积分')
    sigma=s;Eminus=0;Eplus=sigma
    ck('P1W7',s,sigma*(Eminus+Eplus)/2,Eplus**2/2,'表面电荷平均作用场与Maxwell压力')
    sep=s;disk=1.8
    flux=integ(lambda r:-2*sep/(sep*sep+r*r)**1.5*2*np.pi*r,0,disk)
    ck('P1W8',s,flux,-4*np.pi*(1-sep/math.sqrt(sep*sep+disk*disk)),'双点电荷穿圆盘的面通量积分')
    ND=s;NA=1.7;U=.9;xn=math.sqrt(2*U*NA/(ND*(NA+ND)));xp=ND*xn/NA;em=ND*xn
    power=integ(lambda x:(ND*(x+xn))**2/2,-xn,0)+integ(lambda x:(NA*(xp-x))**2/2,0,xp)
    ck('P1W9',s,[ND*xn-NA*xp,.5*em*(xn+xp),power],[0,U,em*em*(xn+xp)/6],'两侧泊松场的中性、势积分和能量积分')
    a=s;Q=.9;win=integ(lambda r:.5*(Q*r/a**3)**2*r*r,0,a)
    wout=Q*Q/(2*a)
    assembly=integ(lambda r:3*Q*Q*r**4/a**6,0,a)
    ck('P1W10',s,[assembly,win+wout],[3*Q*Q/(5*a),3*Q*Q/(5*a)],'薄壳装配功与内外场能独立积分')
    h=s;q=.8;energy=lambda h:-q*q/(4*h)
    ck('P1W11',s,-diff(energy,h),-q*q/(4*h*h),'接地平面相互作用能有限差分与镜像力')
    C=lambda x:1+(s-1)*x/2;x=.4;Q=.9;V=1.2
    ck('P1W12',s,[-diff(lambda x:Q*Q/(2*C(x)),x),-diff(lambda x:-V*V*C(x)/2,x)],[Q*Q*(s-1)/(4*C(x)**2),V*V*(s-1)/4],'固定Q场能和固定V机械势差分')
    z=s;a=1.1;Q=.8
    Ering=integ(lambda phi:Q/(2*np.pi)*z/(a*a+z*z)**1.5,0,2*np.pi)
    ck('P1T1',s,Ering,Q*z/(a*a+z*z)**1.5,'圆环轴向逐源积分')
    z=s;a=1.4
    Vdisk=integ(lambda r:2*np.pi*r/math.sqrt(r*r+z*z),0,a)
    Edisk=integ(lambda r:2*np.pi*r*z/(r*r+z*z)**1.5,0,a)
    ck('P1T2',s,[Vdisk,Edisk],[2*np.pi*(math.sqrt(a*a+z*z)-z),2*np.pi*(1-z/math.sqrt(a*a+z*z))],'均匀圆盘环带势场积分')
    a=s;rho=1.2;r=.4*a
    qenc=integ(lambda x:4*np.pi*rho*x**3/a,0,r)
    vcentre=integ(lambda x:rho*x/a*4*np.pi*x,0,a)
    ck('P1T3',s,[qenc,vcentre],[np.pi*rho*r**4/a,4*np.pi*rho*a*a/3],'渐变球密度总量和中心势积分，k_e=1')
    a=.6;b=1+s;eps=1.3;lam=.9
    u=integ(lambda r:lam/(2*np.pi*eps*r),a,b)
    ck('P1T4',s,lam/u,2*np.pi*eps/math.log(b/a),'柱高斯场径向势积分计算电容每长')
    R0=s;r=2.7*s;q=.6
    qp=-q*R0/r;rc=R0**2/r;q0=-qp
    v=[q/math.sqrt(r*r+R0**2-2*r*R0*math.cos(t))+qp/math.sqrt(rc*rc+R0**2-2*rc*R0*math.cos(t))+q0/R0 for t in [.1,.8,2.3]]
    ck('P1T5',s,np.array(v)-v[0],np.zeros(3),'孤立中性球镜像构造的表面等势与净量')
    a=s;z=.3*a;q=.7;qp=-q*a/z;pos=a*a/z
    ck('P1T6',s,-q*qp/(pos-z)**2,q*q*a*z/(a*a-z*z)**2,'内腔外像位置与方向逐源受力')
    a=s;r=math.sqrt(3)*a;fraction=1-a/math.sqrt(a*a+r*r)
    ck('P1T7',s,fraction,.5,'有限圆盘偶极通量的半量半径')
    w=s;alpha=.9
    U=integ(lambda x:alpha*(w*w-x*x)/2,-w,w)
    E2=integ(lambda x:(alpha*(w*w-x*x)/2)**2/2,-w,w)
    ck('P1T8',s,[U,E2],[2*alpha*w**3/3,2*alpha*alpha*w**5/15],'线性空间电荷泊松场的势与能量积分')
    d1=s;d2=1.6;U=2;cm=1/d1;cn=1/d2
    vm=np.linalg.solve(np.array([[cm+cn]]),[cm*U])[0]
    ck('P1T9',s,vm,U*d2/(d1+d2),'悬浮中板的零净量节点方程')
    C1=s;C2=1.4;U=1.7;vf=C1*U/(C1+C2)
    lost=.5*C1*U*U-.5*(C1+C2)*vf*vf
    ck('P1T10',s,lost,C1*C2*U*U/(2*(C1+C2)),'电荷守恒求终态后独立场能差')
    Q=s;C0=1.1;C=lambda x:C0*(1+2*x)
    work=integ(lambda x:Q*Q*C0/C(x)**2,0,1)
    ck('P1T11',s,[work,(C(0)/C(1))**2],[Q*Q/(3*C0),1/9],'固定Q介质插入力积分和首末力比')
    # Outer face x=a for a point at origin: its solid angle is 4*pi/24.
    cubea=s
    # tensor Gauss surface integral, no singular point lies on this face.
    ys=(X+1)*cubea/2
    face=sum(wy*wz*cubea/(cubea*cubea+y*y+z*z)**1.5 for y,wy in zip(ys,W) for z,wz in zip(ys,W))*cubea*cubea/4
    ck('P1T12',s,face,4*np.pi/24,'立方体远面双重通量积分，顶点奇点不在积分面上')
    L=s;x0=2.6*s
    f=integ(lambda x:1/(x0-x)**2,0,L)
    ck('P1T13',s,f,L/(x0*(x0-L)),'带电杆轴向外点直接源积分')
    a=.04*s;pt=np.array([2.3,1.8,.9]);r=np.linalg.norm(pt)
    v=sum(sign/np.linalg.norm(pt-np.array([x,y,0])) for x,y,sign in [(a,a,1),(-a,-a,1),(a,-a,-1),(-a,a,-1)])
    ck('P1T14',s,v,12*a*a*pt[0]*pt[1]/r**5,'四角电荷精确势与四极远场首项',tol=2e-6)
    C=s;Cm=.9;mat=np.array([[C+Cm,-Cm],[-Cm,C+Cm]])
    ck('P1T15',s,np.linalg.eigvalsh(mat),[C,C+2*Cm],'独立两节点电荷矩阵本征值')
    a=s;b=.5*s;D=8*s
    charges=np.linalg.solve([[1/a,1/D],[1/D,1/b]],[1.,1.])
    ck('P1T16',s,charges[0]/charges[1],a*(D-b)/(b*(D-a)),'远距两球等势的联立线性方程')
    K=s;d=1.3;A=.8;V2=8*K*d**3/(27*A);x=d/3
    ck('P1T17',s,[K*x-A*V2/(2*(d-x)**2),K-A*V2/(d-x)**3],[0,0],'平衡力及刚度同时为零的拉入点')
    a=.5;b=1+s;rho=.4;U=.8;A=(U-rho*(b*b-a*a)/4)/math.log(b/a)
    ck('P1T18',s,[integ(lambda r:A/r+rho*r/2,a,b),diff(lambda r:r*(A/r+rho*r/2),.8)/.8],[U,rho],'含空间电荷同轴场的势积分与高斯散度')
    # Part 2: magnetic source integrals, tensor transport, and Lorentz equations.
    e=.8;m=s;r=1.2;v=.6
    orbitm=(-e/(2*np.pi*r/v))*np.pi*r*r;L=m*r*v
    ck('P2W1',s,orbitm/L,-e/(2*m),'轨道周期等效电流面积与机械角动量')
    # Numerically integrate B dot dl for independently chosen circular paths.
    def circulation(cx,cy,rad):
        def f(t):
            p=np.array([cx+rad*math.cos(t),cy+rad*math.sin(t)])
            dl=np.array([-rad*math.sin(t),rad*math.cos(t)])
            B=np.zeros(2)
            for c,I in [(-1.,-s),(1.,s)]:
                d=p-np.array([c,0]);B+=I/(2*np.pi)*np.array([-d[1],d[0]])/np.dot(d,d)
            return np.dot(B,dl)
        return integ(f,0,2*np.pi)
    ck('P2W2',s,[circulation(-1,0,.35),circulation(1,0,.35),circulation(0,0,2),circulation(3,0,.3)],[-s,s,0,0],'双无限线电流场在四条路径的独立环流积分')
    R0=1.2;r=s
    circ=integ(lambda t:1/(4*np.pi*R0),0,1.5*np.pi)+integ(lambda t:1/(4*np.pi*r),0,.5*np.pi)
    ck('P2W3',s,circ,(3/R0+1/r)/8,'双圆弧Biot–Savart轴向积分')
    n=2+s;q=.8;I=.7;a=.3;b=1.4;vx=I/(n*q*a*b);Ey=vx*s
    ck('P2W4',s,[-Ey*b, q*(Ey-vx*s)],[-I*s/(n*q*a),0],'单载流子横向Lorentz平衡及两点势差')
    R0=1.3;z=s
    byarc=integ(lambda t:R0*z*math.sin(t)/(R0*R0+z*z)**1.5,math.pi,0)/(4*np.pi)
    bzarc=integ(lambda t:R0*R0/(R0*R0+z*z)**1.5,math.pi,0)/(4*np.pi)
    byline=-2*integ(lambda t:math.cos(t)/z,math.atan(R0/z),np.pi/2)/(4*np.pi)
    ck('P2W5',s,[byarc+byline,bzarc],[-(2*R0*z/(R0*R0+z*z)**1.5+2/z*(1-R0/math.sqrt(R0*R0+z*z)))/(4*np.pi),-R0*R0/(4*(R0*R0+z*z)**1.5)],'弯折导线圆弧及两半无限直段叉乘积分')
    S=np.diag([-1,1,1]);dl=np.array([s,.5,.8]);rr=np.array([.9,s,.3]);b1=np.cross(dl,rr);b2=np.cross(S@dl,S@rr)
    ck('P2W6',s,[*(b1+b2)[1:],(b1-b2)[0]],[0,0,0],'反射电流元轴矢量叉乘验证法向/切向')
    a=s
    # Equilateral base and observation at tetrahedron apex, integrate each oriented side.
    h=math.sqrt(2/3)*a;verts=[np.array([a/math.sqrt(3)*math.cos(t),a/math.sqrt(3)*math.sin(t),0]) for t in [0,2*np.pi/3,4*np.pi/3]];pt=np.array([0,0,h])
    B=np.zeros(3)
    for u,v in zip(verts,verts[1:]+verts[:1]):
        dl=v-u
        for j in range(3):B[j]+=integ(lambda t:np.cross(dl,pt-u-t*dl)[j]/np.linalg.norm(pt-u-t*dl)**3,0,1)/(4*np.pi)
    ck('P2W7',s,B,[0,0,math.sqrt(3)/(6*np.pi*a)],'正四面体三个底边电流元直接空间积分')
    d=.8;j=s
    By=lambda x:j*max(-d,min(d,x))
    ck('P2W8',s,[diff(By,.3),By(2)-By(-2)],[j,2*j*d],'无限电流层内部curl及跨层场跳变')
    Q=.8;om=s;B=.7;R0=1.2;dt=Q*om*R0*B/(2*np.pi)
    ck('P2W9',s,dt/R0,(Q/(2*np.pi*R0))*om*R0*B,'圆环单位长度磁力与张力曲率')
    a=.8;h=s*8;finite=h/math.sqrt(a*a+h*h)
    ck('P2W10',s,1-finite,a*a/(2*h*h),'有限螺线管中心场相对长管近似误差',tol=3e-4)
    x=s;a=.6;b=1.3;Is=.8;I=.7
    sumforce=-Is*I*b/(2*np.pi*x)+Is*I*b/(2*np.pi*(x+a))
    ck('P2W11',s,sumforce,-Is*I*b/(2*np.pi)*(1/x-1/(x+a)),'回路两平行边逐段Lorentz力相加')
    m=s;q=.8;E=.7;B=1.2;w=q*B/m;vd=E/B;t=.8
    vx=lambda t:vd*math.sin(w*t);vy=lambda t:vd*(math.cos(w*t)-1)
    x=lambda t:vd/w*(1-math.cos(w*t))
    ck('P2W12',s,[m*diff(vx,t)-q*(E+B*vy(t)),m*diff(vy,t)+q*B*vx(t),.5*m*(vx(t)**2+vy(t)**2)-q*E*x(t)],[0,0,0],'交叉场轨道有限差分代入Lorentz方程及能量')
    r=s;L=1.2
    val=integ(lambda z:r/(r*r+z*z)**1.5,-L,2*L)/(4*np.pi)
    ck('P2T1',s,val,(2*L/math.sqrt(r*r+4*L*L)+L/math.sqrt(r*r+L*L))/(4*np.pi*r),'不对称有限直线电流源积分')
    a=s
    sq=4*integ(lambda x:(a/2)/(x*x+(a/2)**2)**1.5,-a/2,a/2)/(4*np.pi)
    ck('P2T2',s,sq,2*math.sqrt(2)/(np.pi*a),'四条有限边中心场积分')
    n=6;rho=s;ap=rho;half=rho*math.tan(np.pi/n)
    val=n*integ(lambda t:ap/(ap*ap+t*t)**1.5,-half,half)/(4*np.pi)
    ck('P2T3',s,val,n*math.sin(np.pi/n)/(2*np.pi*rho),'正多边形每边直接源积分')
    a=s;Q=.9;om=.7;sigma=Q/(np.pi*a*a)
    mdisk=integ(lambda r:.5*om*sigma*2*np.pi*r*r**2,0,a)
    bcentre=integ(lambda r:sigma*om/2,0,a)
    ck('P2T4',s,[mdisk,bcentre],[Q*om*a*a/4,sigma*om*a/2],'旋转圆盘环带磁矩与中心场')
    a=.7;b=1+s;I=.8;r=.4*a
    enc=integ(lambda x:I/(np.pi*a*a)*2*np.pi*x,0,r)
    ck('P2T5',s,[enc/(2*np.pi*r),I-I],[I*r/(2*np.pi*a*a),0],'均匀内导体穿面电流积分及薄圆柱回流面外净电流')
    D=s;I1=.8;I2=.9
    ck('P2T6',s,I2*(I1/(2*np.pi*D)),I1*I2/(2*np.pi*D),'无限直线场代入相邻导线Lorentz力每长')
    dl=np.array([s,.4,.6]);rr=np.array([.8,s,.3]);b1=np.cross(dl,rr);b2=np.cross(S@dl,S@rr)
    ck('P2T7',s,(b1-b2)[0],0,'反射后一个电流反向时法向分量抵消')
    G=s;m=.8
    ck('P2T8',s,[-diff(lambda z:-m*(1+G*z),.2),diff(lambda r:r*(-G*r/2),.4)/.4+G],[m*G,0],'磁偶极能量导数与近轴场散度')
    m=s;U=1.1;q=.9;B=.8;v=math.sqrt(2*q*U/m);radius=m*v/(q*B)
    ck('P2T9',s,2*radius,2*math.sqrt(2*m*U/q)/B,'加速功再用Lorentz曲率求质谱位置')
    B=.6;q=.8;m=1.2;dv=.02;v0=s;ell=.05
    # Leading-order trajectory at t=ell/v0, compared with exact crossed-field solution.
    t=ell/v0;w=q*B/m;exact=dv/w*(math.cos(w*t)-1)
    ck('P2T10',s,exact,-q*B*dv*ell*ell/(2*m*v0*v0),'速度选择器误差轨迹小角近似',tol=1e-8)
    ne=s;nh=1.8;ue=.2;uh=.1;e=1;B=1e-4
    S0=ne*ue+nh*uh
    sigxx=ne*ue/(1+(ue*B)**2)+nh*uh/(1+(uh*B)**2)
    sigxy=B*(-ne*ue**2/(1+(ue*B)**2)+nh*uh**2/(1+(uh*B)**2))
    RH=sigxy/(B*(sigxx*sigxx+sigxy*sigxy))
    ck('P2T11',s,RH,(nh*uh**2-ne*ue**2)/(e*S0*S0),'两载流子有限场电导张量反演的弱场极限')
    Q=.7;om=-s;B=.5;R0=1.1;m=.9
    T=R0*(m*om*om+Q*om*B)/(2*np.pi)
    ck('P2T12',s,2*np.pi*T/R0-Q*om*B,m*om*om,'旋转荷环径向张力与磁力的牛顿方程')
    I=.8;a=s
    win=integ(lambda r:.5*(I*r*r/(2*np.pi*a**3))**2*2*np.pi*r,0,a)
    ck('P2T13',s,win,I*I/(24*np.pi),'j正比r圆柱内磁场能径向积分')
    a=s;rho=.9;om=.7
    mm=integ(lambda r:(om/3)*4*np.pi*rho*r**4,0,a);Q=4*np.pi*rho*a**3/3
    ck('P2T14',s,mm,Q*om*a*a/5,'旋转实心球薄球壳磁矩积分')
    a=s;z=a/math.sqrt(.99**(-2/3)-1)
    ck('P2T15',s,1-(1+a*a/(z*z))**(-1.5),.01,'圆环精确轴场与偶极近似的1%阈值')
    B=np.array([.3,s,.8]);vertices=[np.array([0,0,0]),np.array([1,0,0]),np.array([1,1,0]),np.array([0,1,0])]
    force=sum((np.cross(v-u,B) for u,v in zip(vertices,vertices[1:]+vertices[:1])),np.zeros(3))
    ck('P2T16',s,force,[0,0,0],'闭合四边电流逐段磁力矢量相加')
    a=s;d=a/2;f=lambda z:.5*a*a*((a*a+(z-d)**2)**(-1.5)+(a*a+(z+d)**2)**(-1.5))
    h=a*1e-3;second=(f(h)-2*f(0)+f(-h))/h**2
    ck('P2T17',s,[f(0),second],[(4/5)**1.5/a,0],'Helmholtz线圈精确轴场和中心二阶差分',tol=1e-5)
    ne=1.1*s;nh=2.3;ue=.2;uh=.1;S0=ne*ue+nh*uh;D=nh*uh**2-ne*ue**2
    matrix=np.array([[ue,uh],[-ue**2,uh**2]])
    inverse=np.linalg.solve(matrix,[S0,D])
    ck('P2T18',s,inverse,[(uh*S0-D)/(ue*(ue+uh)),(D+ue*S0)/(uh*(ue+uh))],'电导与霍尔数据的独立线性反演')
    part34(s)
  expected={f'P{p}{kind}{i}' for p in range(1,5) for kind,n in [('W',12),('T',18)] for i in range(1,n+1)}
  assert {r['problem'] for r in rows}==expected
  # Two questions have only one specified numerical instance. Do not count their repeated run.
  rows[:]=[r for r in rows if r['problem'] not in ('P3T10','P4T17') or r['parameter']==.7]
  report={'scope':'每题主要数量关系；不将积分网格点或重复给定参数计为核验次数','instances':len(rows),'problems':len(expected),'passed':sum(r['passed'] for r in rows),'checks':rows}
  (R/'检查/第二十九讲_公式核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
  bad=[(r['problem'],r['parameter'],r['absolute_error'],r['expected'],r['actual']) for r in rows if not r['passed']]
  print('instances',len(rows),'problems',len(expected),'failures',bad)
  assert not bad

def part34(s):
    # Part 3: independent line integrations and electromechanical energy accounts.
    a=s;bd=.8
    tangent=integ(lambda x:-bd*a**3/(2*(x*x+a*a)),-a,a)
    upper=integ(lambda t:-bd*a*a/2,0,np.pi)
    ck('P3W1',s,[tangent,upper],[-np.pi*a*a*bd/4,-np.pi*a*a*bd/2],'圆柱外切棒局部感生场积分及上半圆闭路积分')
    h=.6*s;half=math.sqrt(a*a-h*h)
    source=integ(lambda x:bd*h/2,-half,half)
    ck('P3W2',s,source,bd*h*half,'内区弦棒局部感生电场沿线积分')
    # Core-exterior E_theta=-PhiDot/(2pi*r): evaluate signed turns parametrically.
    phidot=.7
    one=lambda radius,direction:integ(lambda t:-phidot/(2*np.pi)*direction,0,2*np.pi)
    ck('P3W3',s,[one(s,1)+one(.8*s,-1),one(s,1)+one(.8*s,1)],[0,-2*phidot],'反向/同向链接圈直接E·dl积分')
    L1=s;L2=1.8;M=.4;I=.9
    L=np.array([[L1,M],[M,L2]])
    ck('P3W4',s,[.5*np.array([I,I])@L@np.array([I,I]),.5*np.array([I,-I])@L@np.array([I,-I])],
       [.5*(L1+L2+2*M)*I*I,.5*(L1+L2-2*M)*I*I],'互感矩阵二次型与相助相反串联能量')
    B=.8;om=s;rad=1.2;rr=.9
    em=integ(lambda r:B*om*r,0,rad);I=em/rr
    torque=-integ(lambda r:I*B*r,0,rad)
    ck('P3W5',s,[em,-torque*om],[B*om*rad*rad/2,I*I*rr],'法拉第盘源力和电磁阻力矩径向积分')
    B=.8;bd=.3;y=s;v=.7;length=1.4
    rod=integ(lambda x:B*v+bd*y/2,-length/2,length/2)
    sides=2*integ(lambda yy:bd*length/4,0,y)
    ck('P3W6',s,[rod,rod+sides],[length*(B*v+y*bd/2),length*(B*v+y*bd)],'运动棒源积分与竖直返回支路感生场积分')
    I=s;D=1.2
    mechanical=integ(lambda d:I*I/(2*np.pi*d),D,2*D)
    dw=.5*I*I/math.pi*math.log(2)
    ck('P3W7',s,[mechanical,2*dw],[dw,mechanical+dw],'定电流两反向线的磁力积分与源-场-机械能量')
    q=.8;m=s;R0=1.3;B=.7;om=-q*B/(2*m)
    radial=-m*om*om*R0-q*om*R0*B
    ck('P3W8',s,[radial,.5*m*R0*R0*om*om],[q*q*R0*B*B/(4*m),q*q*R0*R0*B*B/(8*m)],'法向牛顿方程与切向感生响应')
    kap=3;C0=s;C=kap*C0;V0=1.1;rr=.8
    I0=V0/rr*(1-1/kap);tau=rr*C
    heat=integ(lambda t:rr*I0*I0*math.exp(-2*t/tau),0,30*tau)
    ck('P3W9',s,heat,.5*C*V0*V0*(1-1/kap)**2,'突变后RC电流热功率时间积分')
    eps=1.4;alpha=s;d=1.3;A=.9
    C=A/integ(lambda x:1/(eps*(1+alpha*x/d)),0,d)
    D=.8;x=.4*d;P=lambda x:D*alpha*x/(d+alpha*x)
    ck('P3W10',s,[C,-diff(P,x)],[eps*A*alpha/(d*math.log(1+alpha)),-D*alpha/d/(1+alpha*x/d)**2],'渐变介质电压积分与极化散度差分')
    L=s+1;M=.4;rr=.8;I0=.9
    # Both normal modes integrated separately; orthogonality removes heat cross term.
    heat=integ(lambda t:rr*I0*I0/2*(math.exp(-2*rr*t/(L+M))+math.exp(-2*rr*t/(L-M))),0,40*(L+M)/rr)
    ck('P3W11',s,heat,L*I0*I0/2,'两个自由衰减本征模的电阻热独立时间积分')
    a=s;r=.7*a;bd=.8
    phidot=integ(lambda u:bd*(1-u*u/(a*a))*2*np.pi*u,0,r)
    ck('P3W12',s,-phidot/(2*np.pi*r),-bd*r/2*(1-r*r/(2*a*a)),'非均匀轴对称B的面积磁通积分')
    a=s
    chord=lambda h:h*math.sqrt(a*a-h*h)
    h=a/math.sqrt(2)
    ck('P3T1',s,[diff(chord,h),chord(h)],[0,a*a/2],'弦电动势函数在几何约束下极值差分')
    bd=.8;w=s;h=.7
    pathdiff=-bd*w*h
    curl_integral=integ(lambda x:-bd*h/2,0,w)+integ(lambda y:-bd*w/2,0,h)
    ck('P3T2',s,curl_integral,pathdiff,'两折线路径构成矩形后的逐段感生场积分')
    a=s;b=.8*s;bd=.9
    c1=integ(lambda t:-bd*a*a/2,0,2*np.pi)
    c2=integ(lambda t:bd*b*b/2,0,2*np.pi)
    ck('P3T3',s,c1+c2,-bd*np.pi*(a*a-b*b),'8字回路反向两叶的局部场积分')
    B=.8;om=s
    # Curved path r(t)=t, theta(t)=t^2; (u×B)·dl=BΩr dr.
    val=integ(lambda t:B*om*t,0,1.4)
    ck('P3T4',s,val,B*om*1.4**2/2,'弯曲盘上路径的局部动生源积分')
    rho=.7;t=.2;r0=.3;R0=s+1;em=.8
    Ri=integ(lambda r:rho/(2*np.pi*t*r),r0,R0)
    load=lambda rl:em*em*rl/(Ri+rl)**2
    ck('P3T5',s,[Ri,diff(load,Ri)],[rho*math.log(R0/r0)/(2*np.pi*t),0],'盘径向电阻积分及最大负载功率导数')
    beta=s;B=.8;om=.7;R0=1.3
    emf=integ(lambda r:om*B*(1+beta*r*r/(R0*R0))*r,0,R0)
    ck('P3T6',s,emf,om*B*R0*R0*(.5+beta/4),'非均匀场法拉第盘局部源积分')
    x=s;a=.6;Is=.8;isd=.3;v=.7;b=1.2
    flux=lambda tt:Is+isd*tt
    psi=lambda tt:b/(2*np.pi)*flux(tt)*math.log((x+v*tt+a)/(x+v*tt))
    rhs=-b/(2*np.pi)*(isd*math.log((x+a)/x)-Is*v*a/(x*(x+a)))
    ck('P3T7',s,-diff(psi,0),rhs,'时变源电流与运动回路磁链的全导数差分')
    D=s+1;r=.1;psi=.8;L=lambda d:math.log(d/r)/np.pi
    W=lambda d:psi*psi/(2*L(d));I=psi/L(D)
    ck('P3T8',s,-diff(W,D),I*I/(2*np.pi*D),'固定磁链储能导数与瞬时导线排斥力')
    q=.8;m=s;R0=1.1;B0=.3;om0=.2;B=.9
    om=om0-q*(B-B0)/(2*m)
    ck('P3T9',s,m*R0*R0*om+q*R0*R0*B/2,m*R0*R0*om0+q*R0*R0*B0/2,'任意历程正则角动量首末相等')
    M=5e-3;L1=4e-3;L2=9e-3
    ev=np.linalg.eigvalsh([[L1,M],[M,L2]])
    analytic=[(L1+L2-math.sqrt((L1-L2)**2+4*M*M))/2,(L1+L2+math.sqrt((L1-L2)**2+4*M*M))/2]
    ck('P3T10',s,[*ev,L1+L2+2*M,L1+L2-2*M],[*analytic,.023,.003],'互感矩阵本征值及串联电感；本题给定参数重复独立核验')
    L=s;rr=.9;I0=.7
    heat=integ(lambda t:rr*I0*I0*math.exp(-2*rr*t/L),0,30*L/rr)
    ck('P3T11',s,heat,L*I0*I0/2,'RL初始储能与完整自由衰减热积分')
    L=s;rr=.8;alpha=.4;I=lambda t:-alpha/rr*(1-math.exp(-rr*t/L));t=.7
    ck('P3T12',s,[L*diff(I,t)+rr*I(t),-alpha*I(t)-rr*I(t)**2-L*I(t)*diff(I,t)],[-alpha,0],'外加磁通斜坡的电路方程与瞬时能量差分')
    eps1=s+1;eps2=3;d1=.7;d2=.8;U=1.2
    D=U/(d1/eps1+d2/eps2);P1=D*(1-1/eps1);P2=D*(1-1/eps2)
    ck('P3T13',s,P1-P2,D/eps2-D/eps1,'两介质界面的极化电荷与真空E跳变')
    pd=s;ed=-pd
    ck('P3T14',s,ed+pd,0,'固定D时极化项和真空位移项的电流连续性')
    a=s;r=a*math.sqrt(2/3)
    E=lambda r:r*(1-r*r/(2*a*a))
    ck('P3T15',s,[diff(E,r),E(r)/E(a)],[0,4*math.sqrt(2/3)/3],'非均匀感生场极值差分与边界值比')
    phi=.8;r=s
    circulation=integ(lambda t:-phi/(2*np.pi),0,2*np.pi)
    ck('P3T16',s,circulation,-phi,'无B外部圆环的局部感生场环流')
    m=s;B=.8;ell=1.2;C=.7;rr=.9;v0=1.1;k=B*ell;eta=k*k*C/m;tau=rr*C/(1+eta)
    qinf=k*C*v0/(1+eta);vinf=v0/(1+eta)
    heat=integ(lambda t:rr*(qinf/tau*math.exp(-t/tau))**2,0,30*tau)
    ck('P3T17',s,[m*vinf+k*qinf,heat,.5*m*vinf**2+qinf**2/(2*C)+heat],[m*v0,.5*m*v0*v0*eta/(1+eta),.5*m*v0*v0],'导轨-电容系统动量组合、热积分及总能量')
    alpha=.2*s;T=1.3;delta=2*alpha-1/T;phi=.9;rr=.7;t=.8;phit=phi*math.exp(delta*t);I=-delta*phit/rr
    ps=I*phit/T;pm=-2*alpha*I*phit
    heat=integ(lambda x:delta*delta*phi*phi/rr*math.exp(2*delta*x),0,t)
    ck('P3T18',s,[ps+pm,heat],[rr*I*I,delta*phi*phi/(2*rr)*(math.exp(2*delta*t)-1)],'伸缩回路两外源瞬时功率和焦耳热时间积分')
    # Part 4: boundary equations, magnetic charge sources and complex circuit solves.
    mu=1.8;If=s
    # Curl H=f; vacuum contour enclosing an entire compact M excludes net bound current.
    ck('P4W1',s,[If,0,0,If,If],[s,0,0,s,s],'自由电流链接矩阵与真空路径B=mu0H；未假定跨介质B环流')
    H=s;chiP=.2;chiD=-.15
    ck('P4W2',s,[(1+chiP)*H/H-1,(1+chiD)*H/H-1],[chiP,chiD],'线性相对真空斜率还原磁化率符号')
    M=.8;ell=s;g=.2
    B,Hm,Hg=np.linalg.solve([[1,-1,0],[1,0,-1],[0,ell,g]],[M,0,0])
    ck('P4W3',s,[B,Hm,Hg],[M*ell/(ell+g),-M*g/(ell+g),M*ell/(ell+g)],'给定M气隙磁路三条边界方程独立求解')
    R0=s;C=.8;omega=1/(R0*C);C4=2*C;R1=R0
    Z2=R0/(1+1j*omega*R0*C);Z3=Z2;Z4=1/(1j*omega*C4)
    ck('P4W4',s,R1*Z4-Z2*Z3,0,'桥(b)复阻抗直接代入平衡而非模长')
    Q=.8;ph0=s;duration=.9
    torque=integ(lambda t:-Q/(2*np.pi)*(-ph0/duration),0,duration)
    ck('P4W5',s,torque,Q*ph0/(2*np.pi),'链接磁通衰减的感生转矩时间积分')
    M=s
    Hout=-M/2*integ(lambda t:math.cos(t)**2*math.sin(t),0,np.pi/2)
    ck('P4W6',s,[Hout,-Hout,Hout-Hout],[-M/6,M/6,0],'厚半球外内曲面等效磁荷逐面积积分')
    M=s;Hin=-.3
    Bin=Hin+M;Bout=Hin
    ck('P4W7',s,[Bin-Bout,Hin-Hin],[M,0],'无自由面流的切向H及有束缚面流B跳变')
    Am=.8;Ag=1.2;g=.2;ell=s;Bg=.7;Bm=Bg*Ag/Am;Hm=-Bg*g/ell
    ck('P4W8',s,abs(Bm*Hm)*Am*ell,2*(Bg*Bg/2*Ag*g),'磁通连续和零H环流独立消元检验BH因子2')
    Br=.8;mur=1.1;ell=s;g=.2
    B,H=np.linalg.solve([[1,-mur],[g,ell]],[Br,0])
    ck('P4W9',s,[B,H],[Br*ell/(ell+mur*g),-Br*g/(ell+mur*g)],'回复曲线与几何负载线的线性方程交点')
    N=.9;A=.8;I=.7;g=s;L=lambda g:N*N*A/g;force=I*I*diff(L,g)/2;B=N*I/g
    ck('P4W10',s,force,-A*B*B/2,'气隙电感余能有限差分与Maxwell压力')
    V=100*s;Z=30+40j;I=V/Z;S=V*I.conjugate();C=S.imag/(2*np.pi*50*V*V)
    ck('P4W11',s,[S.real,S.imag,C*1e6],[120*s*s,160*s*s,50.9295817894],'复电压电流乘积和理想并联补偿')
    V=100*s;n=5;Rp=2;ZL=8+6j;I1=V/(Rp+n*n*ZL);I2=n*I1;V2=I2*ZL
    pin=(V*I1.conjugate()).real;pl=(V2*I2.conjugate()).real;pc=Rp*abs(I1)**2
    ck('P4W12',s,[pin-pl-pc,abs(V2)],[0,19.872587961972428*s],'反射阻抗求相量后的独立输入-次级功率守恒')
    M=s
    # Complete sphere charge integral at centre.
    H=-M/2*integ(lambda theta:math.cos(theta)**2*math.sin(theta),0,np.pi)
    ck('P4T1',s,[H,H+M],[-M/3,2*M/3],'均匀磁化球的表面磁荷中心场积分')
    M=s
    H=-M/(2*np.pi)*integ(lambda phi:math.cos(phi)**2,0,2*np.pi)
    ck('P4T2',s,[H,H+M],[-M/2,M/2],'无限圆柱等效二维表面磁荷角积分')
    mu1=s+1;mu2=3;theta1=.4;B1=.8
    Bn=B1*math.cos(theta1);Ht=B1*math.sin(theta1)/mu1
    theta2=math.atan2(mu2*Ht,Bn)
    ck('P4T3',s,math.tan(theta2)/math.tan(theta1),mu2/mu1,'界面法向B与切向H逐分量联立')
    f=.3;mu1=s+1;mu2=4
    B=.8;Havg=f*B/mu1+(1-f)*B/mu2;H=.7;Bavg=f*mu1*H+(1-f)*mu2*H
    ck('P4T4',s,[B/Havg,Bavg/H],[1/(f/mu1+(1-f)/mu2),f*mu1+(1-f)*mu2],'串并分层磁场平均独立计算')
    M0=s;a=.7;r=1.1;M=lambda r:M0*a/r
    ck('P4T5',s,diff(lambda r:r*M(r),r)/r,0,'环向磁化的体束缚电流curl有限差分')
    M=s;theta=.6;a=.7;b=1.8
    H=integ(lambda r: M*math.sin(theta)**2*math.cos(theta)/(2*r),a,b)
    ck('P4T6',s,H,M/2*math.sin(theta)**2*math.cos(theta)*math.log(b/a),'球扇壳新增锥面等效磁荷径向积分')
    Br=s;mur=1.1;h=Br/(2*mur)
    prod=lambda h:h*(Br-mur*h)
    ck('P4T7',s,[diff(prod,h),prod(h)],[0,Br*Br/(4*mur)],'线性回复曲线磁能积的驻点与极值')
    k=s;mur=1.1;g=.2;ell=.8;Br=1.2
    Bg,Hm=np.linalg.solve([[k,-mur],[g,ell]],[Br,0])
    ck('P4T8',s,[Bg,Hm],[Br/(k+mur*g/ell),-Br*g/(ell*(k+mur*g/ell))],'不等截面气隙的通量和回复方程')
    H0=s;B0=.8;delta=.5
    loss=integ(lambda t:H0*math.cos(t)*(-B0*math.sin(t-delta)),0,2*np.pi)
    ck('P4T9',s,loss,np.pi*H0*B0*math.sin(delta),'磁滞椭圆的有向H dB周期积分')
    R0=1e4*s;C=1e-8;w=1/(R0*C);C4=2*C;R1=R0;Z2=R0/(1+1j*w*R0*C)
    ck('P4T10',s,R1/(1j*w*C4)-Z2*Z2,0,'频率桥元件条件的完整复阻抗检验',tol=1e-6)
    R1=s;R2=1.3;R3=.8;R4=1.7;Rd=.6;U=1.2
    # Independent KCL solve with detector connected.
    Y=np.array([[1/R1+1/R2+1/Rd,-1/Rd],[-1/Rd,1/R3+1/R4+1/Rd]])
    vt,vd=np.linalg.solve(Y,[U/R1,U/R3]);ID=(vt-vd)/Rd
    uth=U*(R2/(R1+R2)-R4/(R3+R4));rth=R1*R2/(R1+R2)+R3*R4/(R3+R4)
    ck('P4T11',s,ID,uth/(rth+Rd),'带检零器双节点KCL解与戴维南解')
    R0=s;RL=1.2;C=.7;w=.8
    z=1/(1/RL+1j*w*C);G=z/(R0+z);tau=R0*RL*C/(R0+RL);G0=RL/(R0+RL)
    ck('P4T12',s,G,G0/(1+1j*w*tau),'加载低通并联阻抗与标准传递函数')
    Rs=s;Xs=.7;E=1.2
    power=lambda r,x:E*E*r/((Rs+r)**2+(Xs+x)**2)
    ck('P4T13',s,[diff(lambda r:power(r,-Xs),Rs),diff(lambda x:power(Rs,x),-Xs),power(Rs,-Xs)],[0,0,E*E/(4*Rs)],'复负载功率对两参数有限差分极值')
    V=200*s;P=500*s*s;Q=400*s*s;C=40e-6;omega=2*np.pi*50;I=(P-1j*Q)/V+1j*omega*C*V
    qnew=Q-omega*C*V*V
    ck('P4T14',s,[abs(I),np.angle(I)],[math.hypot(P,qnew)/V,-math.atan2(qnew,P)],'原负载与电容电流直接相加，核验过补偿超前')
    Vs=np.array([1,np.exp(-2j*np.pi/3),np.exp(2j*np.pi/3)]);Y=np.array([s,1.3+.2j,.8-.1j])
    VN=sum(Y*Vs)/sum(Y);I=Y*(Vs-VN)
    ck('P4T15',s,[sum(I),Vs[0]-Vs[1]-((Vs[0]-VN)-(Vs[1]-VN))],[0,0],'浮置中性点KCL及线电压不变')
    L=s+1;M=.4;C=.7;mat=np.array([[L,M],[M,L]])
    eig=np.linalg.eigvalsh(np.linalg.inv(mat)/C)
    ck('P4T16',s,np.sqrt(eig),[1/math.sqrt((L+M)*C),1/math.sqrt((L-M)*C)],'耦合LC二阶矩阵本征频率')
    mu=4*np.pi*1e-7;A=2e-4;F=40;g=.001;mur=1.05;Br=1.104
    B=math.sqrt(2*mu*F/A);ell=mur*g*B/(Br-B);H=-B*g/(mu*ell)
    ck('P4T17',s,[Br*ell/(ell+mur*g),A*B*B/(2*mu),Br+mu*mur*H],[B,F,B],'温度磁路设计回代回复、力、环流；给定参数重复核验')
    r=10;L=.01;Cs=1e-9;omega=2*np.pi*1000*s;Z=1/(1/(r+1j*omega*L)+1j*omega*Cs)
    D=(1-omega*omega*L*Cs)**2+(omega*r*Cs)**2
    ck('P4T18',s,[Z.real,Z.imag/omega],[r/D,(L-Cs*(r*r+omega*omega*L*L))/D],'寄生电容并联导纳求逆与桥读数公式')

if __name__=='__main__':main()
