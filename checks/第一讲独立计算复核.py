import sympy as s
import numpy as np
from pathlib import Path
import json,math
OUT=Path(__file__).resolve().parent
x,y,z,a,d,k,Q,q,m,L,g,kap=s.symbols('x y z a d k Q q m L g kap',positive=True)
u,v=s.symbols('u v',positive=True);records=[]
def eq(a,b):
 r=s.simplify(a-b)
 assert r==0,r
 return True
def ck(ident,what,checks):
 assert all(checks);records.append({'id':ident,'关系':what,'通过':True})
x0=d*u/(u+v);Fx=k*q*(u*u/x**2-v*v/(d-x)**2);A=u*u/x0**3+v*v/(d-x0)**3
ck('W1','同号源平衡、轴向恢复系数及横向系数',[eq(Fx.subs(x,x0),0),eq(s.diff(Fx,x).subs(x,x0),-2*k*q*A),eq(s.diff(k*q*y*(u*u/(x0*x0+y*y)**s.Rational(3,2)+v*v/((d-x0)**2+y*y)**s.Rational(3,2)),y).subs(y,0),k*q*A)])
alpha=s.symbols('alpha',positive=True)
ck('W2','异号源两侧的零场位置',[eq((Q/x**2-alpha*Q/(x-d)**2).subs(x,d/(1-s.sqrt(alpha))),0),eq((-Q/x**2+alpha*Q/(d-x)**2).subs(x,-d/(s.sqrt(alpha)-1)),0)])
th=s.symbols('th',positive=True);ss=2*L*s.sin(th);q2=4*m*g*L*L*s.sin(th)**2*s.tan(th)/k
ck('W3','双球精确平衡与小角间距',[eq(k*q2/ss**2,m*g*s.tan(th)),eq(s.limit(q2/th**3,th,0),4*m*g*L**2/k),eq((2*L)**3*k*q*q/(4*m*g*L**2),2*k*q*q*L/(m*g))])
U=k*Q*Q/(2*a)-k*Q*Q/s.Integer(4)*(1/(a+x)+1/(a-x))
ck('W4','三电荷补偿与二阶势能变分',[eq(k*Q*Q/(4*a*a)+k*Q*(-Q/4)/(a*a),0),eq(s.diff(U,x,2).subs(x,0)/2,-k*Q*Q/(2*a**3))])
ck('W5','三角形力投影与中心电荷',[eq(2*k*Q*Q/a**2*s.cos(s.pi/6),s.sqrt(3)*k*Q*Q/a**2),eq(k*Q*(Q/s.sqrt(3))/(a/s.sqrt(3))**2,s.sqrt(3)*k*Q*Q/a**2)])
errs=[]
for N in range(3,26):
 theta=2*np.pi*np.arange(1,N)/N
 r0=np.array([1.,0.]);points=np.stack([np.cos(theta),np.sin(theta)],axis=1)
 delta=r0-points;force=(delta/(np.linalg.norm(delta,axis=1)**3)[:,None]).sum(axis=0)
 exact=np.sum(1/np.sin(np.pi*np.arange(1,N)/N))/4
 errs.append(float(np.max(np.abs(force-[exact,0]))))
ck('W6','正多边形矢量叠加与径向求和',[max(errs)<1e-11])
B=s.symbols('B',positive=True);Fm=-B/x**2-kap*(x-L);Fp=B/x**2-kap*(x-L)
ck('W7','弹簧正负分支的导数、临界点与阈值',[eq(s.diff(Fp,x),-2*B/x**3-kap),eq(s.diff(Fm,x).subs(B,kap*x**2*(L-x)),kap*(2*L/x-3)),eq(s.diff(x*x*(L-x),x).subs(x,2*L/3),0),eq((x*x*(L-x)).subs(x,2*L/3),4*L**3/27)])
r1,r2,C,delta=s.symbols('r1 r2 C delta',positive=True);Ar,Br=s.symbols('Ar Br',positive=True)
ests=s.log(Ar)/s.log(Br)-2
ck('W8','距离指数反演与两个灵敏度系数',[eq(s.diff(ests,Ar),1/(Ar*s.log(Br))),eq(s.diff(ests,Br),-s.log(Ar)/(Br*s.log(Br)**2)),abs(float(s.log(s.Rational(404,100))/s.log(2)-2)-.0143552929770701)<1e-14])
b=s.symbols('b',positive=True);vecs=[s.Matrix([a,b]),s.Matrix([-a,b])]
ck('T1','中垂线合力两个分量',[eq((sum(vecs,s.zeros(2,1))*k*Q*q/(a*a+b*b)**s.Rational(3,2))[0],0),eq((sum(vecs,s.zeros(2,1))*k*Q*q/(a*a+b*b)**s.Rational(3,2))[1],2*k*Q*q*b/(a*a+b*b)**s.Rational(3,2))])
F=k*Q*q*(1/x**2-9/(d-x)**2)
ck('T2','不等源平衡位置与频率',[eq(F.subs(x,d/4),0),eq(-s.diff(F,x).subs(x,d/4)/m,512*k*Q*q/(3*m*d**3))])
ck('T3','异号源右侧零场',[eq((4/x**2-1/(x-d)**2).subs(x,2*d),0)])
ck('T4','接触对称分配与力比值',[eq(((9*q+q)/2)**2/(9*q*q),s.Rational(25,9))])
R=s.symbols('R',positive=True)
Epair=k*Q*(1/(R+x)**2-1/(R-x)**2)+2*k*Q*x/(R**2+x**2)**s.Rational(3,2)
ck('T5','正方形补偿电荷与中心场线性项',[eq((s.sqrt(2)+s.Rational(1,2))/2,(2*s.sqrt(2)+1)/4),eq(s.diff(Epair,x).subs(x,0),-2*k*Q/R**3)])
ck('T6','小角模型的电荷与质量缩放',[eq((s.Integer(2)**s.Rational(2,3))**3,4),eq(s.Integer(8)**(-s.Rational(1,3)),s.Rational(1,2))])
ck('T7','三十度双球精确反演',[eq(q2.subs(th,s.pi/6),m*g*L**2/(s.sqrt(3)*k)),eq(ss.subs(th,s.pi/6),L)])
r,l=s.symbols('r l',positive=True)
Fdip=k*Q*q*(1/(r-l/2)**2-1/(r+l/2)**2)
ck('T8','轴向偶极力精确式与相对误差',[eq(Fdip,2*k*Q*q*l*r/(r*r-l*l/4)**2),eq(s.limit(Fdip*r**3,r,s.oo),2*k*Q*q*l),eq(1-(2*k*Q*q*l/r**3)/Fdip,2*(l/(2*r))**2-(l/(2*r))**4)])
Feqx=k*Q*q*(1/(a+x)**2-1/(a-x)**2);Feqy=2*k*Q*q*y/(a*a+y*y)**s.Rational(3,2)
ck('T9','不同约束的线性运动系数',[eq(s.diff(Feqx,x).subs(x,0),-4*k*Q*q/a**3),eq(s.diff(Feqy,y).subs(y,0),2*k*Q*q/a**3)])
qc=-u*u*v*v/(u+v)**2
ck('T10','三个自由电荷的三条平衡条件',[eq(u*u*v*v/d**2+u*u*qc/x0**2,0),eq(u*u*v*v/d**2+v*v*qc/(d-x0)**2,0),eq((u*u/x**2-v*v/(d-x)**2).subs(x,x0),0)])
crit=Fm.subs(B,4*kap*L**3/27);xc=2*L/3
ck('T11','临界平衡的首个非零合力项',[eq(crit.subs(x,xc),0),eq(s.diff(crit,x).subs(x,xc),0),eq(s.diff(crit,x,2).subs(x,xc)/2,-9*kap/(4*L))])
f,gg,q0=s.symbols('f gg q0',real=True,nonzero=True)
ql=a*a*(f+2*s.sqrt(2)*gg)/(2*k*q0);qr=a*a*(2*s.sqrt(2)*gg-f)/(2*k*q0)
ck('T12','源电荷反演与第三分量预测',[eq(k*q0*(ql-qr)/a**2,f),eq(k*q0*(ql+qr)/(2*s.sqrt(2)*a**2),gg),eq(k*q0*(ql-qr)/(2*s.sqrt(2)*a**2),f/(2*s.sqrt(2)))])
report={'date':'2026-10-21','problems':20,'method':'逐题关键关系的独立符号推导；W6由笛卡尔矢量叠加数值核对N=3至25','scope_limit':'覆盖20题列明的核心计算与稳定性关系，不以参数实例数冒充独立公式数','records':records,'W6_max_abs_error':max(errs)}
(OUT/'记录/第一讲独立计算复核.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('PASS',len(records),'problems')
