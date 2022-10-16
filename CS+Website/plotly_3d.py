import numpy as np
import scipy.ndimage
from skimage import measure
import scipy.ndimage as ndimage
from plotly import __version__
import plotly.figure_factory as ff
from plotly.graph_objs import *
from pydicom import dcmread

# This function resamples voxel space of the 3D image according to the dicom information
def resample(image, new_spacing=[1,1,1]):
    dir = 'CS+Website/data/images/1-01.dcm'
    ds = dcmread(dir)
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

# This function returns verts, faces using marching cubes method
def make_mesh(image, threshold=0.5, step_size=1):

    p = image.transpose(2,1,0)
    p = ndimage.uniform_filter(p, 3) #smoothing out the edges
    
    # marching cubes algorithm: threshold = 0.5 for binary classification
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces

# This function creates a 3D plot using the plotly package
def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = ff.create_trisurf(x=x, y=y, z=z, plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    return fig

# Call this function to combine all the functions above
# INPUT: 3d .npy array
# OUTPUT: plotly 3d figure
def model_3d(file):

    file_used = file
    imgs_to_process = file_used.astype(np.float64)

    imgs_after_resamp, _ = resample(imgs_to_process, [1,1,1])

    v, f = make_mesh(imgs_after_resamp, threshold = 0.5) #350 previously default value
    fig = plotly_3d(v, f)
    return fig

# For testing purposes
if __name__ == "__main__":
    file_used= 'test_data.npy'
    temp = model_3d(file_used)
    print(temp)