import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
from plotly.graph_objs import *
from pydicom import dcmread

def resample(image, new_spacing=[1,1,1]):
    dir = '/Users/easoplee/Desktop/pelvic_project/data/images/images_01/1-01.dcm'
    ds = dcmread(dir)
    #print(f'Slice Thickness: {ds.SliceThickness}')
    #print(f'Pixel Spacing (row, col): ({ds.PixelSpacing[0]}, {ds.PixelSpacing[1]})')
    spacing = np.array([float(ds.SliceThickness), 
                        float(ds.PixelSpacing[0]), 
                        float(ds.PixelSpacing[0])])

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def make_mesh(image, threshold=0.5, step_size=1):

    #print("Transposing surface")
    p = image.transpose(2,1,0)
    p = ndimage.uniform_filter(p, 5)
    
    #print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    #print("Drawing")
    
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = ff.create_trisurf(x=x, y=y, z=z, plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)
    return fig

def plt_3d(verts, faces):
    print("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))

    ax.set_facecolor((0.7,0.7,0.7))
    plt.show()

def model_3d(file):

    file_used = file
    imgs_to_process = np.load(file_used).astype(np.float64)

    #print(f'Shape before resampling: {imgs_to_process.shape}')
    imgs_after_resamp, spacing = resample(imgs_to_process, [1,1,1])
    #print(f'Shape after resampling: {imgs_after_resamp.shape}')

    v, f = make_mesh(imgs_after_resamp, threshold = 0.5) #350 previously default value
    #plt_3d(v, f)
    fig = plotly_3d(v, f)
    return fig

if __name__ == "__main__":
    file_used= 'test_data.npy'
    temp = model_3d(file_used)
    print(temp)