from pathlib import Path
import json,math,numpy as np
A=Path(__file__).resolve().parent;checks=[]
def ck(problem,name,a,b,tol=1e-8):
    a=np.asarray(a);b=np.asarray(b);err=float(np.max(abs(a-b)))
    checks.append(dict(problem=problem,name=name,computed=a.tolist(),reference=b.tolist(),error=err,tolerance=tol,passed=err<=tol))

# The actual red-out / blue-back rectangle has positive signed area.
for a,b,B in [(3,2,.7),(.8,1.7,-.4),(2.3,.9,1.2)]:
    pts=np.array([[0,0],[a,0],[a,b],[0,b]])
    area=.5*sum(np.cross(pts[i],pts[(i+1)%4]) for i in range(4))
    ck('11T10','红去蓝回的有向面积',area,a*b)
    t=np.linspace(0,1,10001)
    # Independent line integrations of A_L and A_s for x-y, y-x, straight.
    integral=np.trapz
    AL=[integral(np.full_like(t,B*a*b),t),0.,integral(B*a*t*b,t)]
    As=[integral(np.full_like(t,B*a*b/2),t),integral(np.full_like(t,-B*a*b/2),t),integral(-B*b*t*a/2+B*a*t*b/2,t)]
    ck('11T10','两规范的三开放路径线积分差',np.array(AL)-As,[B*a*b/2]*3)
    ck('11T10','两种规范闭路积分',[(AL[0]-AL[1]),(As[0]-As[1])],[B*a*b]*2)

# Curves drawn in 4T7: potential contour and field line are orthogonal.
for xi in [.4,.85,1.2]:
    eta=math.log(math.sin(xi))+1.1
    grad=np.array([math.exp(-eta)*math.sin(xi),math.exp(-eta)*math.cos(xi)])
    tangent=np.array([1,math.cos(xi)/math.sin(xi)])
    ck('4T7','场线切向与电场平行',np.cross(tangent,grad),0)
    ck('4T7','场线常数保持',math.exp(-eta)*math.sin(xi),math.exp(-1.1))
    ck('4T7','等势与场线的切向点积',np.dot([1,-math.tan(xi)],tangent),0)

# E(r) maximum: solve nonzero root independently by bisection, then compare plots.
lo,hi=.5,2.
for _ in range(80):
    mid=(lo+hi)/2
    if math.exp(mid)-1-2*mid>0:hi=mid
    else:lo=mid
u=(lo+hi)/2;x=math.sqrt(u);f=lambda v:-math.expm1(-v*v)/v
ck('12T9','非零极值根',u,1.25643,5e-6)
ck('12T9','最大场半径',x,1.12091,5e-6)
ck('12T9','曲线峰值参考线的绘图坐标',2*f(x),1.27635,1e-5)
ck('12T9','轴心磁场归一值',2*math.exp(0)/2,1)
for v in [.3,x,2.7]:
    rr=np.linspace(0,v,20001)
    source=np.trapz(2*rr*np.exp(-rr*rr),rr)/v
    ck('12T9','圆内磁通直接积分形成蓝线',source,f(v),1e-8)
    h=1e-5
    if abs(v-x)<1e-4:ck('12T9','蓝线峰处导数',(f(v+h)-f(v-h))/(2*h),0,2e-6)

# Six surface currents with the stated order M cross outward normal.
Mx,Mz=.7,1.2;M=np.array([Mx,0,Mz])
for n,expected in [([1,0,0],[0,Mz,0]),([-1,0,0],[0,-Mz,0]),([0,1,0],[-Mz,0,Mx]),([0,-1,0],[Mz,0,-Mx]),([0,0,1],[0,-Mx,0]),([0,0,-1],[0,Mx,0])]:
    ck('16W1','六面电流叉乘顺序',np.cross(M,n),expected)

# Refraction directions must use angles from the normal.
ck('18W5','磁感应折射角',math.degrees(math.atan(4*math.tan(math.pi/6))),66.58678,1e-5)
ck('18W5','电场折射角',math.degrees(math.atan(3*math.tan(math.pi/6))),60)
ck('18W5','B图分量比例',np.array([2,.866])/np.array([.5,.866]),[4,1])
ck('18W5','E图分量比例',np.array([.5,.288675])/np.array([.5,.866]),[1,1/3],2e-5)
ck('7W9','电流矢量法向分量连续',[1.1,1.1],[1.1,1.1])
ck('7W9','电流模长比',np.linalg.norm([2.2,1.1])/np.linalg.norm([.55,1.1]),2)
ck('7W9','电场模长比',np.linalg.norm([2.2,1.1])/4/np.linalg.norm([.55,1.1]),.5)
for charge in [.3,-.3]:
    D=2.7;displacement=.4;d1=D/2+displacement;d2=D/2-displacement
    E1=-charge*d2/D;E2=charge*d1/D
    ck('5T17','正负电量的压力合力',.5*(E2*E2-E1*E1),charge*charge*displacement/D)
I1=130/complex(21,40);Zin=complex(21,40)/13;power=10*I1.conjugate()
ck('22W8','功率三角形横、竖边', [power.real,power.imag],[13.37579618,25.47770701])
ck('22W8','功率三角形斜边',abs(power),28.775423,1e-5)
ck('22W8','图中功率有功分支相加',abs(I1)**2*(1+8/13),power.real)
D=lambda x,y:np.array([2+y,0.]);h=1e-5;x0=.4;y0=.3
div=(D(x0+h,y0)[0]-D(x0-h,y0)[0]+D(x0,y0+h)[1]-D(x0,y0-h)[1])/(2*h)
curl=(D(x0+h,y0)[1]-D(x0-h,y0)[1]-D(x0,y0+h)[0]+D(x0,y0-h)[0])/(2*h)
ck('15T2','D沿x而大小随y变的散度',div,0)
ck('15T2','D沿x而大小随y变的旋度',curl,-1)
result=dict(count=len(checks),all_passed=all(x['passed'] for x in checks),scope='本轮改动图的曲线、闭路方向、归一化、叉乘及折射角；一项为一次参数实例比较。',checks=checks)
(A/'新增图解关系复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(result['count'],'checks; all passed',result['all_passed']);assert result['all_passed'],[x for x in checks if not x['passed']]
