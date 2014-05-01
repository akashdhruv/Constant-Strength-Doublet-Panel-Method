import numpy as np
import matplotlib.pyplot as plt
from math import *
import time


# Panel Method code for Constant Strength Doublet Method Using Dirichlet Boundary
# Condition (velocity potential inside the airfoil is zero)

"""Function Definitions"""

# Function NACA to get the airfoil geometry using the definition of NACA airfoils

def NACA(naca,c,n):
    
    H=float(naca[0])/100
    p=float(naca[1])/10
    t1=float(naca[2])
    t2=float(naca[3])
    T=10*t1+t2
    T=T/100

    beta=np.linspace(0,pi,n+1)
    xc=np.zeros(np.size(beta))
    for i in range(n+1):
        xc[i]=c*(1-.5*(1-cos(beta[i])))
    
    thdis=np.zeros(np.size(xc))
    
    for i in range(n+1):
        thdis[i]=5*T*c*(0.2969*sqrt(xc[i]/c)-0.126*xc[i]/c-0.3537*(xc[i]/c)**2 +0.2843*(xc[i]/c)**3-0.1015*(xc[i]/c)**4)
    
    camberline=np.zeros(np.size(beta))
    
    if(p!=0.0 and H!=0.0):
        for i in range(n+1):
            if(xc[i] <= p*c):
                camberline[i]=(H/p**2)*xc[i]*(2*p-xc[i]/c)
            elif(xc[i] > p*c):
                camberline[i]=(H/(1-p)**2)*(c-xc[i])*(1+xc[i]/c-2*p)
    
    xu=np.zeros(np.size(xc))
    xl=np.zeros(np.size(xc))
    zu=np.zeros(np.size(xc))
    zl=np.zeros(np.size(xc))
    tht=np.zeros(np.size(xc))
                
    if(p==0 or H==0):
        xu=xc
        zu=thdis
        xl=xc
        zl=-thdis
    else:
        for i in range(n+1):
            if(xc[i] <= p*c):
                tht[i]=atan((2*H/p)*(-xc[i]/(c*p)+1))
            elif(xc[i] > p*c):
                tht[i]=atan((2*H/(1-p**2))*(p-(xc[i]/c)))
            xu[i]=xc[i]-thdis[i]*sin(tht[i])
            zu[i]=camberline[i]+thdis[i]*cos(tht[i])
            xl[i]=xc[i]+thdis[i]*sin(tht[i])
            zl[i]=camberline[i]-thdis[i]*cos(tht[i])
        
        
    X=np.zeros((n+n+1,1),dtype=float)
    Z=np.zeros((n+n+1,1),dtype=float)
    for i in range(n+1):
        X[i]=xl[i]
        Z[i]=zl[i]
    

    for i in range(n):
        X[n+1+i]=xu[n-i-1]
        Z[n+1+i]=zu[n-i-1]
        
    return X,Z


# Class Freestream to store values of freestream conditions

class freestream:
    def __init__(self,qinf,al):
        self.qinf=qinf # Resultant velocity
        self.alpha=al*pi/180 # Angle of Attack
        self.uinf=qinf*cos(al*pi/180) # X-Velocity
        self.vinf=qinf*sin(al*pi/180) # Y- Velocity


# Class to store panel properties

class Panel:
    def __init__(self,xa,ya,xb,yb):
        self.xa=xa # panel co-ordinate left
        self.ya=ya # panel ordinate left
        self.xb=xb # panel co-ordinate right
        self.yb=yb # panel ordinate right
        self.xc=(xa+xb)/2 # co-ordinate of panel collocation point
        self.yc=(ya+yb)/2 # ordinate of panel collocation point
        self.al=atan2((ya-yb),(xb-xa)) # Panel angle of attack
        self.mu=0 # panel doublet strength
        self.Vt=0 # panel tangential velocity
        self.Cp=0 # panel Cp

# Function PHICD to get value of induced potential by a panel at a given point
# Katz and Plotkin 1991 Page 333 equation number 11.64

def PHICD(mu,xc,yc,panel):
    
    # Function defition PHICD(mu,xc,yc,panel)
    # mu-doublet strength of the panel
    # xc,yc- point at which the induced potential needs to be calculated
    # panel- object which stores panel properties
    # Function returns phi, the potential induced by the panel
    
    xR=cos(panel.al)*(panel.xb-panel.xa)-sin(panel.al)*(panel.yb-panel.ya)
    x=cos(panel.al)*(xc-panel.xa)-sin(panel.al)*(yc-panel.ya)
    y=sin(panel.al)*(xc-panel.xa)+cos(panel.al)*(yc-panel.ya)
    phi=-(mu/(2*pi))*((atan2(y,x-xR))-(atan2(y,x)));
    
    return phi
    


        
        
## Main Function 

tic = time.clock()

N=50 # Number of panels per surface
c=1 # Chord length

# Airfoil type (4-digit)

# naca=[1.1398407488003082, 1.0801909611696667, 1.0, 0.071219861363520209]
# naca=[0,0,1,2]
# naca=[4,4,1,2]
naca=[2,1,0,4]


xp,yp=NACA(naca,c,N) # getting panel co-ordinates


M=len(xp)

panel=np.empty(M,dtype=object) # Defining objects for class Panel

# Storing panel information
for i in range(M):
    if(i==M-1):
        panel[i]=Panel(xp[-1],yp[-1],100000,yp[-1])#Adding wake panel at the end
    else:
        panel[i]=Panel(xp[i],yp[i],xp[i+1],yp[i+1])
    

# Plotting airfoil geometry with wake
plt.figure
plt.plot([p.xb for p in panel],[p.yb for p in panel])
plt.plot([p.xa for p in panel],[p.ya for p in panel])
plt.scatter([p.xc for p in panel],[p.yc for p in panel])
plt.axis("equal")
plt.xlim(-0.05,1.5)
plt.title('Airfoil with Wake panel')
plt.show()


qinf=1.0 # freestream velocity
alpha=5.0# angle of attack

fstream=freestream(qinf,alpha) # fstream object of class freestream


A=np.zeros((M,M),dtype=float) # Array for influence coeffcients

for i in range(M-1):
    for j in range(M):
        if (i==j):
            A[i][j]=0.5
        else:
            A[i][j]=PHICD(1,panel[i].xc,panel[i].yc,panel[j])


# Explicit Kutta Condition
A[M-1][0]=1
A[M-1][M-2]=-1
A[M-1][M-1]=1

RHS=np.zeros((M,1),dtype=float) # Vector for RHS

# RHS=negative of velocity potential due to freestream
for i in range(M-1):
    RHS[i]=-fstream.uinf*panel[i].xc-fstream.vinf*panel[i].yc


mu=np.linalg.solve(A,RHS) # Solving to get doublet strengths


# Loop to calculate Cp and Cl
# Equations Katz and Plotkin 1991 Page 336

Cl=0
for i in range(M-1):
    if i==0:
        R = sqrt((panel[1].xc-panel[0].xc)**2+(panel[1].yc-panel[0].yc)**2);
        panel[i].Vt= (mu[1]-mu[0])/R;
    elif i==M-2:
        R = sqrt((panel[M-2].xc-panel[M-3].xc)**2+(panel[M-2].yc-panel[M-3].yc)**2);
        panel[i].Vt= (mu[M-2]-mu[M-3])/R;
    else:
        R = sqrt((panel[i+1].xc-panel[i-1].xc)**2+(panel[i+1].yc-panel[i-1].yc)**2);
        panel[i].Vt= (mu[i+1]-mu[i-1])/R;
    
    panel[i].Cp=1-((panel[i].Vt/qinf)**2)
    Cl=Cl-panel[i].Vt*R
    
    
# Plotting figures
pxc=np.empty(M-1,dtype=float)
pcp=np.empty(M-1,dtype=float)
for i in range(M-1):
    pxc[i]=panel[i].xc
    pcp[i]=panel[i].Cp
plt.figure()
plt.plot(pxc,pcp)
plt.gca().invert_yaxis()
#plt.title('Coeffcient of Pressure Distribution over the surface')
plt.xlabel('x')
plt.ylabel("C_p")
plt.xlim(0,1)
plt.show()

print "Lift Coeffecient= ",Cl
print "Number of body Panels= ",M-1

toc = time.clock()
print "Time elapsed= ",toc - tic