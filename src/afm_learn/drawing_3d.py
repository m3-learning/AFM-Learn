import numpy as np
from mayavi import mlab
# mlab.init_notebook('x3d')
# mlab.options.offscreen = True
import matplotlib.pyplot as plt

def draw_surface(x, y, z, color):
    if type(z) == int: 
        x_ = np.linspace(x[0], x[1], 10)
        y_ = np.linspace(y[0], y[1], 10) 
        xx, yy = np.meshgrid(x_, y_)
        zz = np.ones((xx.shape[0], xx.shape[1]))*z
    if type(x) == int: 
        y_ = np.linspace(y[0], y[1], 10)    
        z_ = np.linspace(z[0], z[1], 10)    
        yy, zz = np.meshgrid(y_, z_)
        xx = np.ones((yy.shape[0], yy.shape[1]))*x
    if type(y) == int: 
        x_ = np.linspace(x[0], x[1], 10)    
        z_ = np.linspace(z[0], z[1], 10)    
        xx, zz = np.meshgrid(x_, z_)
        yy = np.ones((zz.shape[0], zz.shape[1]))*y
    mlab.mesh(xx, yy, zz, color=color)
     
def draw_box(xrange, yrange, zrange, color):
    draw_surface((xrange[0], xrange[1]), (yrange[0], yrange[1]), zrange[0], color)
    draw_surface((xrange[0], xrange[1]), (yrange[0], yrange[1]), zrange[1], color)
    draw_surface((xrange[0], xrange[1]), yrange[0], (zrange[0], zrange[1]), color)
    draw_surface((xrange[0], xrange[1]), yrange[1], (zrange[0], zrange[1]), color)
    draw_surface(xrange[0], (yrange[0], yrange[1]), (zrange[0], zrange[1]), color)
    draw_surface(xrange[1], (yrange[0], yrange[1]), (zrange[0], zrange[1]), color)
    
def sphere_to_surface(xrange, yrange, z, n, size, layers=1):
    x1 = np.random.randint(xrange[0]+size, xrange[1]-size, size=n)
    y1 = np.random.randint(yrange[0]+size, yrange[1]-size, size=n)
    z1 = np.ones(y1.shape)*z
    mlab.points3d(x1, y1, z1, scale_factor=size)
    
    if layers == 2:
        x2 = np.random.randint(xrange[0]+size, xrange[1]-size, size=n)
        y2 = np.random.randint(yrange[0]+size, yrange[1]-size, size=n)
        x2_new, y2_new = [], []
        for i in range(len(x2)):
            for j in range(len(x1)):
                if x2[i]==x1[j] and y2[i]==y1[j]:
                    x2_new.append(x2[i])
                    y2_new.append(y2[i])
        if len(x2_new) > 1:
            x2_new = np.array(x2_new)
            y2_new = np.array(y2_new)
            z2 = np.ones(y2_new.shape)*(z+1)
            mlab.points3d(x2_new, y2_new, z2, scale_factor=size)