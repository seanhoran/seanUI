import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def variogram(h, nugget=0.0, var=1.0, srange=100.):
    gamma = nugget + var*((3*h)/(2*srange)-(h**3)/(srange**3))
    gamma[h==0]=0.
    gamma[h>srange]=1.0
    return gamma; 

st.title("This is my first app")
st.sidebar.title("this is a sidebar")

x, y, z = np.random.rand(100), np.random.rand(100), np.random.rand(100)

x1,x2 = np.meshgrid(x,x)
y1,y2 = np.meshgrid(y,y)

distmat = ((x1-x2)**2+(y1-y2)**2)**0.5

xy = np.arange(0,1,0.01)
xc,yc = np.meshgrid(xy,xy)
xc = xc.flatten()
yc = yc.flatten()

xc1, xc2 = np.meshgrid(x, xc)
yc1, yc2 = np.meshgrid(y, yc)

rdists = ((xc1-xc2)**2+(yc1-yc2)**2)**0.5

distmat = variogram(distmat)
rdists = variogram(rdists)

invdists = np.linalg.inv(distmat)

weights = np.dot(rdists, invdists)

zc = np.dot(weights,z)

plt.scatter(xc,yc,c=zc)
plt.scatter(x,y,c=z, edgecolor="black")

st.pyplot()







