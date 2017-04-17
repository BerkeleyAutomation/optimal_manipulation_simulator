'''
Encapsulates mesh for grasping operations
Author: Jeff Mahler
'''
import copy
import logging
import IPython
import numpy as np
import Queue
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sklearn.decomposition
import skimage
import sys
import tfx
import time

import camera_params as cp
import colorsys
import obj_file
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
try:
    import pyhull.convex_hull as cvh
except:
    logging.warning('Failed to import pyhull')
import scipy.spatial as ss
import similarity_tf as stf

USE_ALAN=True
try:
    from alan.core import RigidTransform, PointCloud
    from alan.rgbd import CameraIntrinsics
except:
    logging.warning('Failed to import ALAN')
    USE_ALAN=False

C_canonical = np.array([[1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0],
                        [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0],
                        [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0]])

class Mesh3D(object):
    """
    A Mesh is a three-dimensional shape representation
    
    Params:
       vertices:  (list of 3-lists of float)
       triangles: (list of 3-lists of ints)
       normals:   (list of 3-lists of float)
       metadata:  (dictionary) data like category, etc
       pose:      (tfx pose)
       scale:     (float)
       component: (int)
    """
    def __init__(self, vertices, triangles, normals=None, metadata=None, pose=tfx.identity_tf(), scale = 1.0, density=1.0, category='', component=0):
        self.vertices_ = vertices
        self.triangles_ = triangles
        if normals is not None:
            normals = np.array(normals)
            if normals.shape[0] == 3:
                normals = normals.T
        self.normals_ = normals
        self.metadata_ = metadata
        self.pose_ = pose
        self.scale_ = scale
        self.density_ = density
        self.category_ = category
        self.component_ = component
        self.colors_ = None

        # compute mesh properties
        self._compute_bb_center()
        self._compute_centroid()
        #self._compute_com_uniform()
        self.center_of_mass_ = self.bb_center_

    def vertices(self):
        return self.vertices_

    def triangles(self):
        return self.triangles_

    def normals(self):
        if self.normals_ is not None:
            return self.normals_
        return None #"Mesh does not have a list of normals."

    def colors(self):
        if self.colors_ is None:
            return np.ones((len(self.vertices()), 3))
        return self.colors_

    def centroid(self):
        self._compute_centroid()
        return self.vertex_mean_

    def metadata(self):
        if self.metadata_:
            return self.metadata_
        return "No metadata available."

    @property
    def center_of_mass(self):
        # TODO: utilize labelled center of mass if we have it
        return self.center_of_mass_

    @property
    def pose(self):
        return self.pose_

    @pose.setter
    def pose(self, pose):
        self.pose_ = pose

    @property
    def scale(self):
        return self.scale_

    @property
    def mass(self):
        return self.mass_

    @property
    def inertia(self):
        return self.inertia_

    @property
    def density(self):
        return self.density_

    @density.setter
    def density(self, d):
        self.density_ = d
        self.compute_mass()

    @property
    def category(self):
        return self.cateogry_

    @density.setter
    def category(self, c):
        self.category_ = c

    @scale.setter
    def scale(self, scale):
        self.scale_ = scale

    def set_vertices(self, vertices):
        self.vertices_ = vertices

    def set_triangles(self, triangles):
        self.triangles_ = triangles

    def set_normals(self, normals):
        self.normals_ = normals

    def set_colors(self, colors):
        self.colors_ = colors

    def set_metadata(self, metadata):
        self.metadata_ = metadata

    def set_center_of_mass(self, center_of_mass):
        self.center_of_mass_ = center_of_mass
        self.compute_inertia()

    def set_density(self, density):
        self.density_ = density
        self.compute_mass()
        self.compute_inertia()

    def bounding_box(self):
        """ Get the mesh bounding box """
        vertex_array = np.array(self.vertices_)
        min_vertices = np.min(vertex_array, axis=0)
        max_vertices = np.max(vertex_array, axis=0)
        return min_vertices, max_vertices

    def min_coords(self):
        """ Returns the min coordinates """
        vertex_array = np.array(self.vertices_)
        return np.min(vertex_array, axis=0)

    def max_coords(self):
        """ Returns the min coordinates """
        vertex_array = np.array(self.vertices_)
        return np.max(vertex_array, axis=0)

    def bounding_box_mesh(self):
        """Get the mesh bounding box as a mesh."""
        min_vert, max_vert = self.bounding_box()
        xs, ys, zs = zip(max_vert, min_vert)
        vertices = []
        for x in xs:
            for y in ys:
                for z in zs:
                    vertices.append([x, y, z])
        triangles = (np.array([
            [5, 7, 3], [5, 3, 1],
            [2, 4, 8], [2, 8, 6],
            [6, 8, 7], [6, 7, 5],
            [1, 3, 4], [1, 4, 2],
            [6, 5, 1], [6, 1, 2],
            [7, 8, 4], [7, 4, 3],
        ]) - 1).tolist()
        return Mesh3D(vertices, triangles)

    def _compute_bb_center(self):
        """ Get the bounding box center of the mesh  """
        vertex_array = np.array(self.vertices_)
        min_vertices = np.min(vertex_array, axis=0)
        max_vertices = np.max(vertex_array, axis=0)
        self.bb_center_ = (max_vertices + min_vertices) / 2.0

    def _signed_volume_of_tri(self, tri, vertex_array):
        """ Get the bounding box center of the mesh  """
        v1 = vertex_array[tri[0], :]
        v2 = vertex_array[tri[1], :]
        v3 = vertex_array[tri[2], :]

        volume = (1.0 / 6.0) * (v1.dot(np.cross(v2, v3)))
        center = (1.0 / 3.0) * (v1 + v2 + v3)
        return volume, center

    def _covariance_of_tri(self, tri, vertex_array):
        """ Get the bounding box center of the mesh  """
        v1 = vertex_array[tri[0], :]
        v2 = vertex_array[tri[1], :]
        v3 = vertex_array[tri[2], :]

        A = np.zeros([3,3])
        A[:,0] = v1 - self.center_of_mass_
        A[:,1] = v2 - self.center_of_mass_
        A[:,2] = v3 - self.center_of_mass_
        C = np.linalg.det(A) * A.dot(C_canonical).dot(A.T)
        volume = (1.0 / 6.0) * (v1.dot(np.cross(v2, v3)))
        return C, volume

    def _compute_com_uniform(self):
        """ Computes the center of mass using a uniform mass distribution assumption """
        total_volume = 0
        weighted_point_sum = np.zeros([1, 3])
        vertex_array = np.array(self.vertices_)
        for tri in self.triangles_:
            volume, center = self._signed_volume_of_tri(tri, vertex_array)
            weighted_point_sum = weighted_point_sum + volume * center
            total_volume = total_volume + volume
        self.center_of_mass_ = weighted_point_sum / total_volume
        self.center_of_mass_ = np.abs(self.center_of_mass_[0])

    def _compute_centroid(self):
        """ Compute the average of the vertices """
        vertex_array = np.array(self.vertices_)
        self.vertex_mean_ = np.mean(vertex_array, axis=0)

    def compute_mass(self):
        """ Computes the mesh mass. NOTE: Only works for watertight meshes """
        self.mass_ = self.density_ * self.get_total_volume()

    def compute_inertia(self):
        """ Computes the mesh inertia. NOTE: Only works for watertight meshes """
        C = self.get_covariance() 
        self.inertia_ = self.density_ * (np.trace(C) * np.eye(3) - C)

    def compute_normals(self):
        """ Get normals from triangles"""
        vertex_array = np.array(self.vertices_)
        tri_array = np.array(self.triangles_)
        self.normals_ = []
        for i in range(len(self.vertices_)):
            inds = np.where(tri_array == i)
            first_tri = tri_array[inds[0][0],:]
            t = vertex_array[first_tri, :]
            v0 = t[1,:] - t[0,:] 
            v1 = t[2,:] - t[0,:] 
            v0 = v0 / np.linalg.norm(v0)
            v1 = v1 / np.linalg.norm(v1)
            n = np.cross(v0, v1)
            n = n / np.linalg.norm(n)
            self.normals_.append(n.tolist())

        # reverse normal based on alignment with convex hull
        hull = ss.ConvexHull(self.vertices_)
        hull_tris = hull.simplices.tolist()
        hull_vertex_ind = hull_tris[0][0]
        hull_vertex = self.vertices_[hull_vertex_ind]
        hull_vertex_normal = self.normals_[hull_vertex_ind]
        v = np.array(hull_vertex).reshape([1,3])
        n = np.array(hull_vertex_normal)
        ip = (vertex_array - np.tile(hull_vertex, [vertex_array.shape[0], 1])).dot(n)
        if ip[0] > 0:
            self.normals_ = [[-n[0], -n[1], -n[2]] for n in self.normals_]

    def tri_centers(self):
        """ Return a list of the triangle centers as 3D points """
        centers = []
        for tri in self.triangles_:
            v = np.array([self.vertices_[tri[0]], self.vertices_[tri[1]], self.vertices_[tri[2]]])
            center = np.mean(v, axis=0)
            centers.append(center.tolist())
        return centers

    def tri_normals(self, align_to_hull=False):
        """ Return a list of the triangle normals """
        # compute normals
        vertex_array = np.array(self.vertices_)
        tri_array = np.array(self.triangles_)
        v0 = vertex_array[tri_array[:,0],:]
        v1 = vertex_array[tri_array[:,1],:]
        v2 = vertex_array[tri_array[:,2],:]
        n = np.cross(v1 - v0, v2 - v0)
        n = n / np.tile(np.linalg.norm(n, axis=1)[:,np.newaxis], [1,3])
        normals = n.tolist()

        # reverse normal based on alignment with convex hull
        if align_to_hull:
            tri_centers = self.tri_centers()
            hull = ss.ConvexHull(tri_centers)
            hull_tris = hull.simplices.tolist()
            hull_vertex_ind = hull_tris[0][0]
            hull_vertex = tri_centers[hull_vertex_ind]
            hull_vertex_normal = normals[hull_vertex_ind]
            v = np.array(hull_vertex).reshape([1,3])
            n = np.array(hull_vertex_normal)
            ip = (np.array(tri_centers) - np.tile(hull_vertex, [np.array(tri_centers).shape[0], 1])).dot(n)
            if ip[0] > 0:
                normals = [[-n[0], -n[1], -n[2]] for n in normals]
        return normals

    def get_total_volume(self):
        total_volume = 0
        vertex_array = np.array(self.vertices_)
        for tri in self.triangles_:
            volume, center = self._signed_volume_of_tri(tri, vertex_array)            
            total_volume = total_volume + volume

        # can get negative volume when tris are flipped, so auto correct assuming that mass should have been postive
        if total_volume < 0:
            logging.debug('Volume was negative. Flipping sign, but mesh may be degenerate')
            total_volume = -total_volume
        return total_volume

    def get_covariance(self):
        C_sum = np.zeros([3,3])
        vertex_array = np.array(self.vertices_)
        for tri in self.triangles_:
            C, volume = self._covariance_of_tri(tri, vertex_array)            
            C_sum = C_sum + C
        return C_sum
    
    def principal_dims(self):
        """ Return the mesh principal dimensions """
        vertex_array = np.array(self.vertices_)
        min_vertex_coords = np.min(self.vertices_, axis=0)
        max_vertex_coords = np.max(self.vertices_, axis=0)
        vertex_extent = max_vertex_coords - min_vertex_coords
        return vertex_extent

    def project_binary(self, camera_params):
        '''
        Project the triangles of the mesh into a binary image which is 1 if the mesh is
        visible at that point in the image and 0 otherwise.
        The image is assumed to be taken from a camera with specified params at
        a given relative pose to the mesh.
        Params:
           camera_params: CameraParams object
           camera_pose: 4x4 pose matrix (camera in mesh basis)
        Returns:
           PIL binary image (1 = mesh projects to point, 0 = does not)
        '''
        vertex_array = np.array(self.vertices_)
        camera_pose = camera_params.pose().matrix()
        R = camera_pose[:3,:3]
        t = camera_pose[:3,3:]
        t_rep = np.tile(t, [1, 3]) # replicated t vector for fast transformations
        
        height = camera_params.get_height()
        width = camera_params.get_width()
        img_shape = [int(height), int(width)]
        fill_img = Image.fromarray(np.zeros(img_shape).astype(np.uint8)) # init grayscale image
        img_draw = ImageDraw.Draw(fill_img)

        # project each triangle and fill in image
        for f in self.triangles_:
            verts_mesh_basis = np.array([vertex_array[f[0],:], vertex_array[f[1],:], vertex_array[f[2],:]])

            # transform to camera basis
            verts_cam_basis = R.dot(verts_mesh_basis.T) + t_rep
            # project points to the camera imaging plane and snap to image borders
            verts_proj, valid = camera_params.project(verts_cam_basis)
            verts_proj[0, verts_proj[0,:] < 0] = 0
            verts_proj[1, verts_proj[1,:] < 0] = 0
            verts_proj[0, verts_proj[0,:] >= width]  = width - 1
            verts_proj[1, verts_proj[1,:] >= height] = height - 1

            # fill in area on image
            img_draw.polygon([tuple(verts_proj[:,0]), tuple(verts_proj[:,1]), tuple(verts_proj[:,2])], fill=255)
        return fill_img

    def project_depth(self, camera_params, cull=True):
        '''
        Project the triangles of the mesh into a binary image which is 1 if the mesh is
        visible at that point in the image and 0 otherwise.
        The image is assumed to be taken from a camera with specified params at
        a given relative pose to the mesh.
        Params:
           camera_params: CameraIntrinsics object
           camera_pose: 4x4 pose matrix (camera in mesh basis)
        Returns:
           PIL binary image (1 = mesh projects to point, 0 = does not)
        '''
        start = time.time()
        vertex_array = np.array(self.vertices_)
        tri_array = np.array(self.triangles_)

        if cull:
            tri_norms = np.array(self.tri_normals())
            valid_ind = np.where(tri_norms[:,2] < 0)[0]
            tri_array = tri_array[valid_ind,:]

        point_cloud = PointCloud(vertex_array.T,
                                 frame=camera_params.frame)
        verts_proj_ind = camera_params.project(point_cloud)
        verts_proj = verts_proj_ind.data.T

        height = camera_params.height
        width = camera_params.width
        depth_im = np.zeros([height, width])

        u0 = verts_proj[tri_array[:,0],:]
        u1 = verts_proj[tri_array[:,1],:]
        u2 = verts_proj[tri_array[:,2],:]
        d0 = vertex_array[tri_array[:,0],2]
        d1 = vertex_array[tri_array[:,1],2]
        d2 = vertex_array[tri_array[:,2],2]

        v0 = (u1 - u0).astype(np.float32)
        v1 = (u2 - u0).astype(np.float32)

        d00 = np.sum(v0 * v0, axis=1)
        d01 = np.sum(v0 * v1, axis=1)
        d11 = np.sum(v1 * v1, axis=1)
        denoms = d00 * d11 - d01 * d01

        x_coords = np.c_[u0[:,0], u1[:,0], u2[:,0]]
        y_coords = np.c_[u0[:,1], u1[:,1], u2[:,1]]

        for i in range(x_coords.shape[0]):
            # check valid denom
            denom = denoms[i]
            if denom == 0:
                continue

            # compute pixels withing the projected triangle
            pixels = skimage.draw.polygon(np.array([u0[i,1], u1[i,1], u2[i,1]]),
                                          np.array([u0[i,0], u1[i,0], u2[i,0]]))                                         
            rows = np.r_[pixels[0], u0[i,1], u1[i,1], u2[i,1]]
            columns = np.r_[pixels[1], u0[i,0], u1[i,0], u2[i,0]]
            num_pixels = rows.shape[0]

            # interpolate depth for each pixel
            for k in range(num_pixels):
                u = np.array([columns[k], rows[k]])
                if u[0] >= 0 and u[0] < width and u[1] >= 0 and u[1] < height:
                    # compute barycentric coords
                    v2 = (u - u0[i,:]).astype(np.float32)
                    d20 = v2.dot(v0[i,:])
                    d21 = v2.dot(v1[i,:])
                    b = (d11[i] * d20 - d01[i] * d21) / denom
                    c = (d00[i] * d21 - d01[i] * d20) / denom
                    a = 1.0 - b - c

                    # interpolate depth
                    d = a * d0[i] + b * d1[i] + c * d2[i]

                    # assign depth if closer than current value
                    if depth_im[u[1], u[0]] == 0 or d < depth_im[u[1], u[0]]:
                        depth_im[u[1], u[0]] = d

        """
        plt.figure()
        plt.imshow(depth_im, cmap=plt.cm.gray)
        plt.show()
        """
        return depth_im

    def remove_unreferenced_vertices(self):
        '''
        Clean out vertices (and normals) not referenced by any triangles.
        '''
        vertex_array = np.array(self.vertices_)
        num_v = vertex_array.shape[0]

        # fill in a 1 for each referenced vertex
        reffed_array = np.zeros([num_v, 1])
        for f in self.triangles_:
            reffed_array[f[0]] = 1
            reffed_array[f[1]] = 1
            reffed_array[f[2]] = 1

        # trim out vertices that are not referenced
        reffed_v_old_ind = np.where(reffed_array == 1)
        reffed_v_old_ind = reffed_v_old_ind[0]
        reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1 # counts number of reffed v before each ind

        try:
            self.vertices_ = vertex_array[reffed_v_old_ind, :].tolist()
            if self.normals_ is not None:
                normals_array = np.array(self.normals_)
                self.normals_ = normals_array[reffed_v_old_ind, :].tolist()
        except IndexError:
            return False

        # create new face indices
        new_triangles = []
        for f in self.triangles_:
            new_triangles.append([reffed_v_new_ind[f[0]], reffed_v_new_ind[f[1]], reffed_v_new_ind[f[2]]] )
        self.triangles_ = new_triangles
        self._compute_centroid()
        return True

    def image_to_3d_coords(self):
        '''
        Flip x and y axes (if created from image this might help)
        '''
        if len(self.vertices_) > 0:
            vertex_array = np.array(self.vertices_)
            new_vertex_array = np.zeros(vertex_array.shape)
            new_vertex_array[:,0] = vertex_array[:,1]
            new_vertex_array[:,1] = vertex_array[:,0]
            new_vertex_array[:,2] = vertex_array[:,2]
            self.vertices_ = new_vertex_array.tolist()
            return True
        else:
            return False
    
    def center_vertices_avg(self):
        '''
        Centers vertices at average vertex
        '''
        vertex_array = np.array(self.vertices_)
        centroid = np.mean(vertex_array, axis = 0)
        vertex_array_cent = vertex_array - centroid
        self.vertices_ = vertex_array_cent.tolist()

    def center_vertices_bb(self):
        '''
        Centers vertices at center of bounding box
        '''
        vertex_array = np.array(self.vertices_)
        min_vertex = np.min(vertex_array, axis = 0)
        max_vertex = np.max(vertex_array, axis = 0)
        centroid = (max_vertex + min_vertex) / 2
        vertex_array_cent = vertex_array - centroid
        self.vertices_ = vertex_array_cent.tolist()

    def normalize_vertices(self):
        '''
        Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
        centroid and the principal axes are aligned with the vertical Z, Y, and X axes.
        Returns:
           Nothing. Modified the mesh in place (for now)
        '''
        self.center_vertices_avg()
        vertex_array_cent = np.array(self.vertices_)
        
        # find principal axes
        pca = sklearn.decomposition.PCA(n_components = 3)
        pca.fit(vertex_array_cent)

        # count num vertices on side of origin wrt principal axes
        comp_array = pca.components_
        norm_proj = vertex_array_cent.dot(comp_array.T)
        opposite_aligned = np.sum(norm_proj < 0, axis = 0)
        same_aligned = np.sum(norm_proj >= 0, axis = 0)
        pos_oriented = 1 * (same_aligned > opposite_aligned) # trick to turn logical to int
        neg_oriented = 1 - pos_oriented
        
        # create rotation from principal axes to standard basis
        target_array = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) # Z+, Y+, X+
        target_array = target_array * pos_oriented + -1 * target_array * neg_oriented
        R = np.linalg.solve(comp_array, target_array)
        R = R.T

        # rotate vertices, normals and reassign to the mesh
        vertex_array_rot = R.dot(vertex_array_cent.T)
        vertex_array_rot = vertex_array_rot.T
        self.vertices_ = vertex_array_rot.tolist()
        self.center_vertices_bb()

        if self.normals_:
            normals_array = np.array(self.normals_)
            normals_array_rot = R.dot(normals_array.T)
            self.normals_ = normals_array_rot.tolist()

    def copy(self):
        # WARNING: this only copies vertices and triangles
        vertices = list(self.vertices())
        triangles = list(self.triangles())
        return Mesh3D(vertices, triangles)

    @staticmethod
    def max_edge_length(tri, vertices):
        v0 = np.array(vertices[tri[0]])
        v1 = np.array(vertices[tri[1]])
        v2 = np.array(vertices[tri[2]])
        max_edge_len = max(np.linalg.norm(v1-v0), max(np.linalg.norm(v1-v0), np.linalg.norm(v2-v1)))
        return max_edge_len

    def subdivide(self, min_triangle_length = None):
        '''Returns a copy of the current mesh but subdivided by one iteration'''
        new_mesh = self.copy()
        new_vertices = new_mesh.vertices()
        old_triangles = new_mesh.triangles()
        
        new_triangles = []
        triangle_index_mapping = {}
        tri_queue = Queue.Queue()
        
        for j, triangle in enumerate(old_triangles):
            tri_queue.put((j, triangle))
            triangle_index_mapping[j]  = []

        while not tri_queue.empty():
            tri_index_pair = tri_queue.get()
            j = tri_index_pair[0]
            triangle = tri_index_pair[1]

            if min_triangle_length is None or Mesh3D.max_edge_length(triangle, new_vertices) > min_triangle_length:
                t_vertices = np.array([new_vertices[i] for i in triangle])
                edge01 = 0.5 * (t_vertices[0,:] + t_vertices[1,:])
                edge12 = 0.5 * (t_vertices[1,:] + t_vertices[2,:])
                edge02 = 0.5 * (t_vertices[0,:] + t_vertices[2,:])

                i_01 = len(new_vertices)
                i_12 = len(new_vertices)+1
                i_02 = len(new_vertices)+2
                new_vertices.append(edge01)
                new_vertices.append(edge12)
                new_vertices.append(edge02)
            
                for triplet in [[triangle[0], i_01, i_02], [triangle[1], i_12, i_01], [triangle[2], i_02, i_12], [i_01, i_12, i_02]]:
                    tri_queue.put((j, triplet))

            else:
                new_triangles.append(triangle)
                triangle_index_mapping[j].append(len(new_triangles)-1)

        new_mesh.set_vertices(new_vertices)
        new_mesh.set_triangles(new_triangles)
        return new_mesh, triangle_index_mapping

    def transform(self, T):
        # TODO: support mesh frame referencing
        if USE_ALAN and isinstance(T, RigidTransform):
            vertex_cloud = PointCloud(np.array(self.vertices_).T, frame=T.from_frame)
            vertex_cloud_tf = T * vertex_cloud
            vertices = vertex_cloud_tf.data.T.tolist()
        elif isinstance(T, stf.SimilarityTransform3D):
            vertex_cloud = np.array(self.vertices_).T
            vertex_cloud_tf = T.apply(vertex_cloud)
            vertices = vertex_cloud_tf.T.tolist()            
        return Mesh3D(vertices, self.triangles_)

    def rescale_vertices(self, min_scale):
        '''
        Rescales the vertex coordinates so that the minimum dimension (X, Y, Z) is exactly min_scale
        Params:
           min_scale: (float) the scale of the min dimension
        Returns:
           Nothing. Modified the mesh in place (for now)
        '''
        vertex_array = np.array(self.vertices_)
        min_vertex_coords = np.min(self.vertices_, axis=0)
        max_vertex_coords = np.max(self.vertices_, axis=0)
        vertex_extent = max_vertex_coords - min_vertex_coords

        # find minimal dimension
        min_dim = np.where(vertex_extent == np.min(vertex_extent))
        min_dim = min_dim[0][0]

        # compute scale factor and rescale vertices
        scale_factor = min_scale / vertex_extent[min_dim] 
        vertex_array = scale_factor * vertex_array
        self.vertices_ = vertex_array.tolist()
        self._compute_centroid()

    def rescale(self, scale_factor):
        '''
        Rescales the vertex coordinates by scale factor
        Params:
           scale_factor: (float) the scale factor
        Returns:
           Nothing. Modified the mesh in place (for now)
        '''
        vertex_array = np.array(self.vertices_)
        scaled_vertices = scale_factor * vertex_array
        self.vertices_ = scaled_vertices.tolist()

    def convex_hull(self):
        """ Returns the convex hull of a mesh as a new mesh """
        hull = ss.ConvexHull(self.vertices_)
        hull_tris = hull.simplices.tolist()
        # TODO do normals properly...
        cvh_mesh = Mesh3D(self.vertices_, hull_tris)#, self.normals_)
        cvh_mesh.remove_unreferenced_vertices()
        return cvh_mesh

    def make_image(self, filename, rot):
        proj_img = self.project_binary(cp, T)
        file_root, file_ext = os.path.splitext(filename)
        proj_img.save(file_root + ".jpg")                

        oof = obj_file.ObjFile(filename)
        oof.write(self)

    def visualize(self, color=(0.5, 0.5, 0.5), style='surface', opacity=1.0):
        """ Plots visualization """
        vertex_array = np.array(self.vertices_)
        surface = mv.triangular_mesh(vertex_array[:,0], vertex_array[:,1], vertex_array[:,2], self.triangles_, representation=style,
                                     color=color, opacity=opacity)
        return surface

    def visualize_segments(self, tri_labels, style='surface'):
        """ Plots visualization """
        vertex_array = np.array(self.vertices_)
        tri_array = np.array(self.triangles_)
        num_segments = np.max(tri_labels)+1
        color_delta = 1.0 / num_segments
        colors = []
        for j in range(num_segments):
            colors.append(colorsys.hsv_to_rgb(j*color_delta, 0.9, 0.9))
            tris_segment = tri_array[np.where(tri_labels==j)[0],:]
            surface = mv.triangular_mesh(vertex_array[:,0], vertex_array[:,1], vertex_array[:,2], tris_segment, representation=style,
                                         color=colors[-1])

    def create_json_metadata(self):
        return {
            'mass': self.mass,
            'category': self.category
        }

    def to_hdf5(self, h):
        """ Add mesh to hdf5 group """
        h.create_dataset('triangles', data=np.array(self.triangles_))
        h.create_dataset('vertices', data=np.array(self.vertices_))
        if self.normals_ is not None:
            h.create_dataset('normals', data=np.array(self.normals_))

    """
    def num_connected_components(self):
        vert_labels = np.linspace(0, len(self.vertices_)-1, num=len(self.vertices_)).astype(np.uint32)
        for t in self.triangles_:
            vert_labels[t[1]] = vert_labels[t[0]]
            vert_labels[t[2]] = vert_labels[t[0]]
    """                   

def test_mass_inertia():
    filename = '/home/jmahler/Libraries/hacd/20151014-09d5310-sbs51-YCB_Black_and_Decker-3d/textured-0008192.obj'
    com = np.array([-0.030715, 0.024351, 0.023568])
    of = obj_file.ObjFile(filename)
    m = of.read()
    m.set_center_of_mass(com)
    
    print 'MASS = ', m.mass
    print 'INERTIA'
    print m.inertia

if __name__ == '__main__':
    test_mass_inertia()
