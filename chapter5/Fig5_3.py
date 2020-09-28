#reproduction of fig 5.3
#2020/9/17
#by flyingairplane
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
N=50
x=np.random.uniform(size=(N,))
x.sort()

#for natural cubic spline
def dkX(x,k,K):
    return ((x - k > 0) * 1.0 * np.power(x - k, 3)-(x - K > 0) * 1.0 * np.power(x - K, 3))/(K-k)
def truncated_power_basis(x,knots):
    K=len(knots)
    H=np.zeros((len(x),K))
    H[:,0]=1
    H[:,1]=x
    for k in range(1,K-2+1):
        H[:,k+1]=dkX(x,knots[k-1],knots[K-1])-dkX(x,knots[K-2],knots[K-1])
    return H

#global linear
h0=np.ones((N,))
h1=x
Hgl=np.vstack((h0,h1)).T
Eta=np.linalg.inv(Hgl.T.dot(Hgl))
Vargl=np.diag(Hgl.dot(Eta).dot(Hgl.T))

#global cubic poly
Hgcp=np.zeros((N,4))
Hgcp[:,0]=np.ones((N,))
Hgcp[:,1]=x
Hgcp[:,2]=x**2
Hgcp[:,3]=np.power(x,3)
Etagcp=np.linalg.inv(Hgcp.T.dot(Hgcp))
Vargcp=np.diag(Hgcp.dot(Etagcp).dot(Hgcp.T))

#cubic spline 2knots
epsilon1=0.33
epsilon2=0.66
Hcs=np.zeros((N,6))
Hcs[:,0]=np.ones((N,))
Hcs[:,1]=x
Hcs[:,2]=x**2
Hcs[:,3]=np.power(x,3)
Hcs[:,4]=(x-epsilon1>0)*1.0*np.power(x-epsilon1,3)
Hcs[:,5]=(x-epsilon2>0)*1.0*np.power(x-epsilon2,3)
Etacs=np.linalg.inv(Hcs.T.dot(Hcs))
Varcp=np.diag(Hcs.dot(Etacs).dot(Hcs.T))

#natural cubic spline 6-knots
knots=np.linspace(0.1,0.9,6)
Hncs=truncated_power_basis(x,knots)
Etancs=np.linalg.inv(Hncs.T.dot(Hncs))
Varncs=np.diag(Hncs.dot(Etancs).dot(Hncs.T))

plt.style.use('ggplot')
fig,ax=plt.subplots()
ax.plot(x,Vargl,'-',marker='o',markersize=3,label='global linear')
ax.plot(x,Vargcp,'-',marker='o',markersize=3,label='global cubic poly')
ax.plot(x,Varcp,'-',marker='o',markersize=3,label='cubic spline 2-knots')
ax.plot(x,Varncs,'-',marker='o',markersize=3,label='natural cubic spline 6-knots')
ax.set_xlabel('X')
ax.set_ylabel('pointwise variance')
ax.legend(loc='upper center')
plt.show()