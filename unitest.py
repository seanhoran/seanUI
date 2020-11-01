import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def variogram(h, nugget=0.0, var=1.0, srange=100.):
    gamma = nugget + var*((3*h)/(2*srange)-(h**3)/(2*srange**3))
    gamma[h==0]=0.
    gamma[h>srange]=1.0
    return gamma; 

@st.cache()

st.title("Kriging Example")
st.write("Small proof of concept APP created by Sean Horan")
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

nugget = st.slider(label="Nugget", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="knugget")
var = 1.0 - nugget
srange = st.slider(label="Range", min_value=0.0, max_value=5.0, value=2.0, step=0.5, key="krange")


distmat = variogram(distmat, nugget=nugget, var=var, srange=srange)
rdists = variogram(rdists, nugget=nugget, var=var, srange=srange)

invdists = np.linalg.inv(distmat)
weights = np.dot(rdists, invdists)

kvar = 1 - (np.sum(distmat) - np.sum(weights*rdists, axis=1))

zc = np.dot(weights,z) + (1.-np.sum(weights, axis=1))*np.average(z)

st.write("Variogram")

fig, ax = plt.subplots()
lags = np.arange(0., 5.0, 0.01)
lvars = variogram(lags, nugget=nugget, var=var, srange=srange)
ax.plot(lags, lvars, "-r")
ax.axis(xmin=0, xmax=5.0)
st.pyplot(fig)

st.write("Kriged Grade")

fig, ax = plt.subplots()
ax.scatter(xc,yc,c=zc)
ax.scatter(x,y,c=z, edgecolor="black")
st.pyplot(fig)

st.write("Kriging Variance")

fig, ax = plt.subplots()
ax.scatter(xc,yc,c=kvar)
ax.scatter(x,y,c=z, edgecolor="black")
st.pyplot(fig)

st.write("Sum of weights")

fig, ax = plt.subplots()
im=ax.scatter(xc,yc,c=np.sum(weights, axis=1))
fig.colorbar(im)
st.pyplot(fig)


st.image("Lyceum Presentation.jpg", width=700)



