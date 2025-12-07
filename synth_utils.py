from __future__ import division
import numpy as np 
from ransac import fit_plane_ransac
from sys import modules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import mayavi.mlab as mym


class LUT_RGB(object):
    """
    RGB LUT for Mayavi glyphs.
    """
    def __create_8bit_rgb_lut__():
        xl = np.mgrid[0:256, 0:256, 0:256]
        lut = np.vstack((xl[0].reshape(1, 256**3),
                         xl[1].reshape(1, 256**3),
                         xl[2].reshape(1, 256**3),
                         255 * np.ones((1, 256**3)))).T
        return lut.astype('int32')
    __lut__ = __create_8bit_rgb_lut__()

    @staticmethod
    def rgb2scalar(rgb):
        """
        return index of RGB colors into the LUT table.
        rgb : nx3 array
        """
        return 256**2*rgb[:,0] + 256*rgb[:,1] + rgb[:,2]

    @staticmethod
    def set_rgb_lut(myv_glyph):
        """
        Sets the LUT of the Mayavi glyph to RGB-LUT.
        """
        lut = myv_glyph.module_manager.scalar_lut_manager.lut
        lut._vtk_obj.SetTableRange(0, LUT_RGB.__lut__.shape[0])
        lut.number_of_colors = LUT_RGB.__lut__.shape[0]
        lut.table = LUT_RGB.__lut__


def plot_xyzrgb(xyz,rgb,show=False):
    """
    xyz : nx3 float
    rgb : nx3 uint8

    Plots a RGB-D point-cloud in mayavi.
    """
    rgb_s = LUT_RGB.rgb2scalar(rgb)
    pts_glyph = mym.points3d(xyz[:,0],xyz[:,1],xyz[:,2],
                             rgb_s,mode='point')
    LUT_RGB.set_rgb_lut(pts_glyph)
    if show:
        mym.view(180,180)
        mym.orientation_axes()
        mym.show()

def visualize_plane(pt,plane,show=False):
    """
    Visualize the RANSAC PLANE (4-tuple) fit to PT (nx3 array).
    Also draws teh normal.
    """
    # # plot the point-cloud:
    if show and mym.gcf():
        mym.clf()

    # plot the plane:
    plane_eq = '%f*x+%f*y+%f*z+%f'%tuple(plane.tolist())
    m,M = np.percentile(pt,[10,90],axis=0)
    implicit_plot(plane_eq, (m[0],M[0],m[1],M[1],m[2],M[2]))

    # plot surface normal:
    mu = np.percentile(pt,50,axis=0)
    mu_z = -(mu[0]*plane[0]+mu[1]*plane[1]+plane[3])/plane[2]
    mym.quiver3d(mu[0],mu[1],mu_z,plane[0],plane[1],plane[2],scale_factor=0.3)

    if show:
        mym.view(180,180)
        mym.orientation_axes()
        mym.show(True)

def implicit_plot(expr, ext_grid, Nx=11, Ny=11, Nz=11,
                 col_isurf=(50/255, 199/255, 152/255)):
    """
    Function to plot algebraic surfaces described by implicit equations in Mayavi
 
    Implicit functions are functions of the form
 
        `F(x,y,z) = c`
 
    where `c` is an arbitrary constant.
 
    Parameters
    ----------
    expr : string
        The expression `F(x,y,z) - c`; e.g. to plot a unit sphere,
        the `expr` will be `x**2 + y**2 + z**2 - 1`
    ext_grid : 6-tuple
        Tuple denoting the range of `x`, `y` and `z` for grid; it has the
        form - (xmin, xmax, ymin, ymax, zmin, zmax)
    fig_handle : figure handle (optional)
        If a mayavi figure object is passed, then the surface shall be added
        to the scene in the given figure. Then, it is the responsibility of
        the calling function to call mlab.show().
    Nx, Ny, Nz : Integers (optional, preferably odd integers)
        Number of points along each axis. It is recommended to use odd numbers
        to ensure the calculation of the function at the origin.
    """
    xl, xr, yl, yr, zl, zr = ext_grid
    x, y, z = np.mgrid[xl:xr:eval('{}j'.format(Nx)),
                       yl:yr:eval('{}j'.format(Ny)),
                       zl:zr:eval('{}j'.format(Nz))]
    scalars = eval(expr)
    src = mym.pipeline.scalar_field(x, y, z, scalars)
    cont1 = mym.pipeline.iso_surface(src, color=col_isurf, contours=[0],
                                      transparent=False, opacity=0.8)
    cont1.compute_normals = False # for some reasons, setting this to true actually cause
                                  # more unevenness on the surface, instead of more smooth
    cont1.actor.property.specular = 0.2 #0.4 #0.8
    cont1.actor.property.specular_power = 55.0 #15.0


def ensure_proj_z(plane_coeffs, min_z_proj):
    a,b,c,d = plane_coeffs
    if np.abs(c) < min_z_proj:
        s = ((1 - min_z_proj**2) / (a**2 + b**2))**0.5
        coeffs = np.array([s*a, s*b, np.sign(c)*min_z_proj, d])
        assert np.abs(np.linalg.norm(coeffs[:3])-1) < 1e-3
        return coeffs
    return plane_coeffs

def isplanar(xyz,sample_neighbors,dist_thresh,num_inliers,z_proj):
    """
    Checks if at-least FRAC_INLIERS fraction of points of XYZ (nx3)
    points lie on a plane. The plane is fit using RANSAC.

    XYZ : (nx3) array of 3D point coordinates
    SAMPLE_NEIGHBORS : 5xN_RANSAC_TRIALS neighbourhood array
                       of indices into the XYZ array. i.e. the values in this
                       matrix range from 0 to number of points in XYZ
    DIST_THRESH (default = 10cm): a point pt is an inlier iff dist(plane-pt)<dist_thresh
    FRAC_INLIERS : fraction of total-points which should be inliers to
                   to declare that points are planar.
    Z_PROJ : changes the surface normal, so that its projection on z axis is ATLEAST z_proj.

    Returns:
        None, if the data is not planar, else a 4-tuple of plane coeffs.
    """
    frac_inliers = num_inliers/xyz.shape[0]
    dv = -np.percentile(xyz,50,axis=0) # align the normal to face towards camera
    max_iter = sample_neighbors.shape[-1]
    plane_info =  fit_plane_ransac(xyz,neighbors=sample_neighbors,
                            z_pos=dv,dist_inlier=dist_thresh,
                            min_inlier_frac=frac_inliers,nsample=20,
                            max_iter=max_iter) 
    if plane_info != None:
        coeff, inliers = plane_info
        coeff = ensure_proj_z(coeff, z_proj)
        return coeff,inliers
    else:
        return #None


class DepthCamera(object):
    """
    Camera functions for Depth-CNN camera.
    """
    f = 520

    @staticmethod
    def plane2xyz(center, ij, plane):
        """
        converts image pixel indices to xyz on the PLANE.

        center : 2-tuple
        ij : nx2 int array
        plane : 4-tuple

        return nx3 array.
        """
        ij = np.atleast_2d(ij)
        n = ij.shape[0]
        ij = ij.astype('float')
        xy_ray = (ij-center[None,:]) / DepthCamera.f
        z = -plane[2]/(xy_ray.dot(plane[:2])+plane[3])
        xyz = np.c_[xy_ray, np.ones(n)] * z[:,None]
        return xyz

    @staticmethod
    def depth2xyz(depth):
        """
        Convert a HxW depth image (float, in meters)
        to XYZ (HxWx3).

        y is along the height.
        x is along the width.
        """
        H,W = depth.shape
        xx,yy = np.meshgrid(np.arange(W),np.arange(H))
        X = (xx-W/2) * depth / DepthCamera.f
        Y = (yy-H/2) * depth / DepthCamera.f
        return np.dstack([X,Y,depth.copy()])

    @staticmethod
    def overlay(rgb, depth):
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = depth - np.min(depth)
        vmax  = np.max(depth)
        if not np.isfinite(vmax) or vmax < 1e-8:
            vmax = 1.0
        depth = depth / vmax
        depth = np.clip(depth, 0.0, 1.0)
        depth = (255 * depth).astype('uint8')
        return np.dstack([rgb[:, :, 0], depth, rgb[:, :, 1]])


def get_texture_score(img,masks,labels):
    """
    gives a textureness-score
    (low -> less texture, high -> more texture) for each mask.
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img.astype('float32')/255.0
    img = sim.filters.gaussian_filter(img,sigma=1)  
    G = np.clip(np.abs(sim.filters.laplace(img)),0,1)

    tex_score = []
    for l in labels:
        ts = np.sum(G[masks==l].flat)/np.sum((masks==l).flat)
        tex_score.append(ts)
    return np.array(tex_score)

import numpy as np
_EPS = 1e-8

def _safe_unit(v, axis=None):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.maximum(n, _EPS)
    out = v / n
    out[~np.isfinite(out)] = 0.0
    return out

def ssc(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(v)
    if n < _EPS:
        return np.zeros((3, 3), dtype=np.float64)
    v = v / n
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]], dtype=np.float64)

def rot3d(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64).reshape(-1)
    v2 = np.asarray(v2, dtype=np.float64).reshape(-1)
    if not np.isfinite(v1).all():
        v1 = np.nan_to_num(v1, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(v2).all():
        v2 = np.nan_to_num(v2, nan=0.0, posinf=0.0, neginf=0.0)

    v1 = _safe_unit(v1)
    v2 = _safe_unit(v2)

    v3 = np.cross(v1, v2)
    s  = np.linalg.norm(v3)
    c  = float(np.dot(v1, v2))
    c  = np.clip(c, -1.0, 1.0)

    if s < _EPS:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if np.abs(v1[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = _safe_unit(np.cross(v1, axis))
            Vx = ssc(axis)
            return np.eye(3) + 2 * Vx.dot(Vx)

    Vx = ssc(v3)
    Vx = Vx / max(s, _EPS)
    return np.eye(3) + (s * Vx) + ((1.0 - c) * Vx.dot(Vx))

def unrotate2d(pts):
    """
    Стабильная версия: центрируем, считаем ковариацию и её SVD.
    При вырожденности возвращаем I (без поворота).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"unrotate2d expects (N,2), got {pts.shape}")

    # уберём мусор
    if not np.isfinite(pts).all():
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)

    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ковариация 2x2
    C = (X.T @ X) / max(len(pts) - 1, 1)
    if not np.isfinite(C).all():
        C = np.zeros_like(C)

    try:
        # SVD более стабилен, чем eig на симм. матрице при шумах
        U, S, Vt = np.linalg.svd(C)
        R = U   # 2x2 "ориентация"
        # страховка: если почему-то NaN/Inf — вернём I
        if not np.isfinite(R).all():
            R = np.eye(2)
    except np.linalg.LinAlgError:
        R = np.eye(2)

    # Возвращаем «распрямляющую» матрицу
    return R