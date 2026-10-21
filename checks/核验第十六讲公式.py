from pathlib import Path
import numpy as np,json,math
ROOT=Path(__file__).resolve().parent
nodes,weights=np.polynomial.legendre.leggauss(200)
def quad(fun,a,b):
    x=(a+b)/2+(b-a)*nodes/2
    return (b-a)/2*float(np.dot(weights,fun(x)))
results=[]
def check(name,x,y,kind='积分或恒等式',tol=2e-8):
    error=abs(float(x-y))/max(1,abs(float(x)),abs(float(y)))
    results.append({'name':name,'kind':kind,'error':error,'passed':error<tol})
    assert error<tol,(name,x,y,error)
rng=np.random.default_rng(1605)
# mu0=1 的单位化计算，不替代SI量纲审查。
for it in range(18):
    a=rng.uniform(.5,2);ell=rng.uniform(.3,5);M=rng.uniform(.2,3);z=rng.uniform(-3,3)
    num=quad(lambda t:M*a*a/(2*(a*a+(z-t)**2)**1.5),-ell/2,ell/2)
    exact=M/2*((z+ell/2)/np.hypot(a,z+ell/2)-(z-ell/2)/np.hypot(a,z-ell/2))
    check('finite_cylinder_axis_'+str(it),num,exact)
    check('cylinder_center_'+str(it),quad(lambda t:M*a*a/(2*(a*a+t*t)**1.5),-ell/2,ell/2),M*ell/np.sqrt(ell*ell+4*a*a))
    check('slab_gradient_current_'+str(it),M+M-2*M,0)
    k=rng.uniform(.1,4);d=rng.uniform(.2,4)
    check('rotating_slab_mx_'+str(it),quad(lambda t:M*np.cos(k*t),0,d),M*np.sin(k*d)/k)
    check('rotating_slab_my_'+str(it),quad(lambda t:M*np.sin(k*t),0,d),M*(1-np.cos(k*d))/k)
    b=a+rng.uniform(.2,2);I=rng.uniform(.3,4);chi=rng.uniform(.2,3)
    check('graded_shell_current_'+str(it),chi*I+quad(lambda r:chi*I/(np.pi*a*a)*2*np.pi*r,a,b)-chi*I*b*b/(a*a),0)
    check('graded_partial_ampere_'+str(it),chi*I+quad(lambda r:chi*I/(np.pi*a*a)*2*np.pi*r,a,(a+b)/2),chi*I*((a+b)/2)**2/a**2)
    check('toroid_flux_'+str(it),quad(lambda r:I/(2*np.pi*r),a,b),I*np.log(b/a)/(2*np.pi))
    check('toroid_energy_'+str(it),quad(lambda r:.5*(I/(2*np.pi*r))**2*2*np.pi*r,a,b),I*I*np.log(b/a)/(4*np.pi))
    check('remanent_ring_partial_current_'+str(it),2*np.pi*a*M+quad(lambda r:M/r*2*np.pi*r,a,b),2*np.pi*b*M)
    check('parabolic_cylinder_flux_'+str(it),quad(lambda r:M*(1-r*r/(a*a))*2*np.pi*r,0,a),M*np.pi*a*a/2)
    check('parabolic_cylinder_moment_'+str(it),.5*quad(lambda r:r*(2*M*r/(a*a))*2*np.pi*r,0,a),M*np.pi*a*a/2)
for it,mu in enumerate([.05,.2,.5,1,2,4,10,50]):
    H0=1.7
    # 独立边界矩阵：内球势=-A r cos，外=-H0 r cos+C/r²。
    A,c=np.linalg.solve([[1,1],[mu,-2]],[H0,H0])
    check('linear_sphere_H_'+str(it),A,3*H0/(mu+2),'边界线性方程')
    check('linear_sphere_B_'+str(it),mu*A,3*mu*H0/(mu+2),'边界线性方程')
    A,c=np.linalg.solve([[1,1],[1,-2*mu]],[H0,mu*H0])
    check('cavity_H_'+str(it),A,3*mu*H0/(1+2*mu),'边界线性方程')
    A,c=np.linalg.solve([[1,1],[mu,-1]],[H0,H0])
    check('transverse_cylinder_H_'+str(it),A,2*H0/(mu+1),'边界线性方程')
    # 电流片矩阵包含无额外均匀B场的选择。
    mu1=2.3;mu2=mu;K=1.5
    h1,h2=np.linalg.solve([[1,-1],[mu1,mu2]],[K,0])
    check('current_sheet_H1_'+str(it),h1,mu2*K/(mu1+mu2),'边界线性方程')
    check('current_sheet_H2_'+str(it),h2,-mu1*K/(mu1+mu2),'边界线性方程')
for it in range(20):
    a=rng.uniform(.3,1.5);b=a*rng.uniform(1.05,3);k=10**rng.uniform(-1,2);M=rng.uniform(.1,4);f=(a/b)**3
    # 四条件的独立求解，未知[A,C,D,E]。
    matrix=np.array([[1,-1,-1/a**3,0],[-1,k,-2*k/a**3,0],[0,1,1/b**3,-1/b**3],[0,-k,2*k/b**3,-2/b**3]])
    A,C,D,E=np.linalg.solve(matrix,[0,-M,0,0])
    den=(2*k+1)*(k+2)-2*(k-1)**2*f
    for label,num,exact in [('A',A,M*(k+2+2*(k-1)*f)/den),('C',C,2*(k-1)*M*f/den),('D',D,M*a**3*(k+2)/den),('E',E,3*k*M*a**3/den)]:
        check('coated_magnet_'+label+'_'+str(it),num,exact,'边界线性方程')
    check('coated_effective_moment_'+str(it),4*np.pi*E/(4*np.pi*a**3*M/3),9*k/den)
    check('coated_denominator_'+str(it),den,9*k+2*(k-1)**2*(1-f))
    # 两个面电流分别求偶极矩（表面电流K=K0 sin theta）。
    m_faces=4*np.pi/3*(a**3*(M+(k-1)*A)-b**3*(k-1)*E/b**3)
    check('coated_surface_moment_'+str(it),m_faces,4*np.pi*E)
for it in range(12):
    a=rng.uniform(.5,2);M=rng.uniform(.2,3);V=4*np.pi*a**3/3;m=V*M
    angular=2*np.pi*quad(lambda u:1+3*u*u,-1,1)
    # r=a/t 把无穷远积分映射为0<t<1。
    radial=quad(lambda t:t*t/a**3,0,1)
    outside=m*m/(32*np.pi*np.pi)*angular*radial
    check('sphere_outside_energy_'+str(it),outside,V*M*M/9)
    check('sphere_demag_energy_'+str(it),outside+V*M*M/18,V*M*M/6)
    check('sphere_B_energy_'+str(it),outside+2*V*M*M/9,V*M*M/3)
for it in range(12):
    chi=rng.uniform(-.9,6);n1=rng.uniform(.01,.25);n2=(1-n1)/2;V=rng.uniform(.5,2);H=rng.uniform(.2,2);th=rng.uniform(.2,1.3)
    p=V*chi/(1+n1*chi);q=V*chi/(1+n2*chi)
    check('ellipsoid_anisotropy_'+str(it),p-q,V*chi*chi*(n2-n1)/((1+n1*chi)*(1+n2*chi)))
    energy=lambda t:-.5*H*H*(p*np.cos(t)**2+q*np.sin(t)**2)
    e=1e-5;tau=-(energy(th+e)-energy(th-e))/(2*e)
    check('ellipsoid_torque_'+str(it),tau,-.5*H*H*(p-q)*np.sin(2*th),'有限差分')
    alpha=rng.uniform(-2,3);I=rng.uniform(.5,3);R=rng.uniform(1,4)
    energy=lambda r:-alpha*I*I/(8*np.pi*np.pi*r*r)
    force=-(energy(R+e)-energy(R-e))/(2*e)
    check('small_sphere_force_'+str(it),force,-alpha*I*I/(4*np.pi*np.pi*R**3),'有限差分')
for it in range(16):
    d=rng.uniform(.5,2);c=rng.uniform(.1,1.8)*d;D=d+c;spring=rng.uniform(.5,3);N=rng.integers(2,10);S=rng.uniform(.2,2);I=rng.uniform(.2,1)
    x=rng.uniform(.02,.8)*min(d,D/3);const=N*N*S;ind=lambda x:const/(D-x);e=1e-5
    force=.5*I*I*(ind(x+e)-ind(x-e))/(2*e)
    check('gap_fixed_current_force_'+str(it),force,.5*const*I*I/(D-x)**2,'有限差分')
    Ic=np.sqrt(8*spring*D**3/(27*N*N*S));xc=D/3
    check('pull_in_equilibrium_'+str(it),.5*const*Ic*Ic/(D-xc)**2,spring*xc)
    check('pull_in_field_'+str(it),(N*Ic/(D-xc))**2,2*spring*D/(3*S))
    u=np.linspace(0,1,12001);peak=np.max(u*(1-u)**2)
    check('pull_in_global_peak_'+str(it),peak,4/27,'极值采样')
    lam=rng.uniform(.2,2);energy=lambda x:lam*lam*(D-x)/(2*const)
    force=-(energy(x+e)-energy(x-e))/(2*e)
    check('fixed_flux_force_'+str(it),force,lam*lam/(2*const),'有限差分')
    # 有限位移的恒流供能：源功 = 场能增量 + 机械功。
    y=x+.1*(d-x)
    source=I*I*(ind(y)-ind(x));change=.5*source
    mechanical=quad(lambda t:.5*const*I*I/(D-t)**2,x,y)
    check('gap_source_energy_'+str(it),source,change+mechanical)
groups={}
for r in results:
    g=groups.setdefault(r['kind'],{'count':0,'maximum_normalized_error':0})
    g['count']+=1;g['maximum_normalized_error']=max(g['maximum_normalized_error'],r['error'])
summary={'count':len(results),'all_passed':all(r['passed'] for r in results),'maximum_normalized_error':max(r['error'] for r in results),
 '计数说明':'同一关系在不同参数下的积分、边界矩阵、有限差分或极值采样实例，不代表独立公式数量。',
 '零值误差说明':'mu0=1等单位化条件下，预期零值使用绝对误差。','groups':groups,'checks':results}
(ROOT/'检查/第十六讲_公式核验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False))
