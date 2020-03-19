
import nestle
import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize , stats


d=np.loadtxt('Data.txt')
data=d.transpose()

x=data[0]
y=data[1]
yerr=data[2]



def hypo1(x,a):
    	return a*x**0

def hypo2(x,a,m):
    	return a*np.exp(-m*x)
        
def hypo3(x,a,m):
    	return a*np.cos(m*x + 3.0*np.pi/4.0)


def log_likelihood1(P,DATA):
    y_fit=hypo1(DATA[0],P[0])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))
def log_likelihood2(P,DATA):
    y_fit=hypo2(DATA[0],P[0],P[1])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))
def log_likelihood3(P,DATA):
    y_fit=hypo3(DATA[0],P[0],P[1])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))

def chi2_val1(P,DATA):
        sigma=DATA[2]
        y_fit=hypo1(DATA[0],P[0])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)
def chi2_val2(P,DATA):
        sigma=DATA[2]
        y_fit=hypo2(DATA[0],P[0],P[1])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)
def chi2_val3(P,DATA):
        sigma=DATA[2]
        y_fit=hypo3(DATA[0],P[0],P[1])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#degrees of freedom
def dof_val(P,DATA):
  return len(DATA[0]) - len(P)


#chi squared likelihood function
def chi2L(chi2,dof):
  return stats.chi2(dof).pdf(chi2)


#bayesian priors
def prior_transform_1(P1,data):
    return P1*(np.max(y)-np.min(y)) + np.min(y)

def nestle1(DATA):
    f = lambda P1: log_likelihood1(P1, DATA)
    prior = lambda P1: prior_transform_1(P1,DATA)
    res = nestle.sample(f, prior, 1, method='multi',
                    npoints=2000)
    return res

 
def prior_transform_2(P2,data):    
    return np.array([P2[0]*(np.max(y)-np.min(y)) + np.min(y), P2[1]*25.0] )

def nestle2(DATA):
    f = lambda P2: log_likelihood2(P2, DATA)
    prior = lambda P2: prior_transform_2(P2,DATA)
    res = nestle.sample(f, prior, 2, method='multi',
                    npoints=2000)
    return res

def prior_transform_3(P3,data):    
    return np.array([P3[0]*(np.max(y)-np.min(y)) + np.min(y), P3[1]*70.0] )

def nestle3(DATA):
    f = lambda P3: log_likelihood3(P3, DATA)
    prior = lambda P3: prior_transform_3(P3,DATA)
    res = nestle.sample(f, prior, 2, method='multi',
                    npoints=2000)
    return res

#freq,aic,bic tests
def analysis(P1,P2,P3,DATA):
    chi2_1,chi2_2,chi2_3= chi2_val1(P1,DATA),chi2_val2(P2,DATA),chi2_val3(P3,DATA)
    dof1,dof2,dof3 =  dof_val(P1,DATA), dof_val(P2,DATA), dof_val(P3,DATA)
    
    aic1=-2*log_likelihood1(P1,DATA) + 2
    aic2=-2*log_likelihood2(P2,DATA) + 2*2
    aic3=-2*log_likelihood3(P3,DATA) + 2*2
    del_aic2= np.abs(aic2-aic1)
    del_aic3= np.abs(aic3-aic1)
    
    bic1=-2*log_likelihood1(P1,DATA) + np.log(len(DATA[0]))
    bic2=-2*log_likelihood2(P2,DATA) + 2*np.log(len(DATA[0]))
    bic3=-2*log_likelihood3(P3,DATA) + 2*np.log(len(DATA[0]))
    del_bic2= np.abs(bic2-bic1)
    del_bic3= np.abs(bic3-bic1)    

    
    print("newtonian : alpha=",'%.4f'%P1[0])
    print("yukawa : alpha= ",'%.4f'%P2[0]," , m= ",'%.4f'%P2[1])
    print("oscillating : alpha= ",'%.4f'%P3[0]," , m= ",'%.4f'%P3[1])

    print("\nchi2 value/dof\nnewtonian :", '%.4f'%chi2_1,'/',dof1,"\nyukawa :", '%.4f'%chi2_2,'/',dof2,"\noscillating:",'%.4f'%chi2_3,'/',dof3)
    print("\nchi2 likelihoods\nnewtonian :", '%.4f'%chi2L(chi2_1,dof1),"\nyukawa :", '%.4f'%chi2L(chi2_2,dof2),"\noscillating:", '%.4f'%chi2L(chi2_3,dof3))
    print("\nAIC values","\nnewtonian:",'%.2f'%aic1,"\nyukawa:",'%.2f'%aic2,"\noscillating:",'%.2f'%aic3)
    print("\nBIC values","\nnewtonian:",'%.2f'%bic1,"\nyukawa:",'%.2f'%bic2,"\noscillating:",'%.2f'%bic3)

    print("\nyukawa vs newtonian")
    d=np.abs(chi2_2-chi2_1)
    print("difference in chi square values = ",'%.4f'%d)
    p=stats.chi2(dof1-dof2).sf(d)
    print ("p value=",'%.2f'%p)
    print("Confidence level : ",'%.2f'%stats.norm.isf(p),'\u03C3')
    print("delta AIC = ",'%.2f'%del_aic2)
    print("delta BIC = ",'%.2f'%del_bic2)
    
    print("\noscillating vs newtonian")
    d=np.abs(chi2_3-chi2_1)
    print("difference in chi square values = ",'%.2f'%d)
    p=stats.chi2(dof1-dof3).sf(d)
    print ("p value=",'%.4f'%p)
    print("Confidence level : ",'%.2f'%stats.norm.isf(p),'\u03C3')
    print("delta AIC = ",'%.2f'%del_aic3)
    print("delta BIC = ",'%.2f'%del_bic3)

#bayesian test
def bayesian():
    z1=nestle1(data)
    z2=nestle2(data)
    z3=nestle3(data)
    
    print("\nbayesian")
    print("newtonian ","logz=",z1.logz,"; a=",np.mean(z1.samples))
    print("yukawa ","logz=",z2.logz,"; a=",np.mean(z2.samples[:,0]),"; m=",np.mean(z2.samples[:,1]))
    print("oscillating ","logz=",z3.logz,"; a=",np.mean(z3.samples[:,0]),"; m=",np.mean(z3.samples[:,1]))
    print("bayes factor yukawa-newtonian=",np.exp(z2.logz-z1.logz))
    print("bayes factor osc-newt",np.exp(z3.logz-z1.logz))
    print("bayes factor osc-yukawa",np.exp(z3.logz-z2.logz))


def plot(P1,P2,P3):
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,8))
    
    
    plt.subplot(111)
    plt.xscale("log")
    p1 = np.linspace(data[0].min(),data[0].max(),10000)   
    plt.plot(p1, hypo1(p1,P1[0]),label='Newtonian',color='black')
    plt.plot(p1, hypo2(p1,P2[0],P2[1]),label='Yukawa',color='dodgerblue')
    plt.plot(p1, hypo3(p1,P3[0],P3[1]),linewidth=1.5,label='oscillating',color='indianred')
    plt.scatter(data[0],data[1],c='grey',s=25)
    plt.grid(color='w')
    plt.legend(loc='upper right',title='parametrization',fontsize=20,title_fontsize=20)
    plt.tick_params(axis='both',labelsize=20)
    plt.errorbar(data[0],data[1],yerr = data[2],fmt='none',alpha=0.6,c='black')
    plt.xlabel("r (mm)",fontsize=20)
    plt.ylabel("Residual torque (fN.m)",fontsize=20)
    plt.savefig("fig.png")
    
#-==========================================================================-======
def main(): 
    g1=np.array([0])
    g2=np.array([0,20])
    g3=np.array([0,65])
    
    bnd1=[(np.min(y),np.max(y))]
    bnd2=[(np.min(y),np.max(y)),(0,25)]
    bnd3=[(np.min(y),np.max(y)),(0,70)]
    
    P1 = optimize.minimize(chi2_val1, g1,args=data, bounds=bnd1, method='SLSQP')
    P2 = optimize.minimize(chi2_val2, g2,args=data, bounds=bnd2, method='SLSQP')
    P3 = optimize.minimize(chi2_val3, g3,args=data, bounds=bnd3, method='SLSQP')
    
    
    plot(P1.x,P2.x,P3.x)
    
    analysis(P1.x,P2.x,P3.x,data)

    bayesian()
