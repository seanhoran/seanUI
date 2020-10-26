import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def variogram(h, nugget=0.0, var=1.0, srange=100.):
    gamma = nugget + var*((3*h)/(2*srange)-(h**3)/(srange**3))
    gamma[h==0]=0.
    gamma[h>srange]=1.0
    return gamma; 

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

zc = np.dot(weights,z)

fig, ax = plt.subplots()

ax.scatter(xc,yc,c=zc)
ax.scatter(x,y,c=z, edgecolor="black")

st.pyplot(fig)

#url = "https://github.com/seanhoran/seanUI/blob/main/Lyceum Presentation.pdf"
#html = '<embed src="' + url +  '" width="800px" height="2100px" />'
#html = '<iframe src="https://docs.google.com/gview?url=' + url + '" style="width:600px; height:500px;" frameborder="0"></iframe>'
#html = '<embed src="' + url + '" type="application/pdf"   height="700px" width="500">'

#st.write(html)

#st.components.v1.html(html, width=None, height=None, scrolling=False)

st.image("Lyceum Presentation.jpg", width=700)



