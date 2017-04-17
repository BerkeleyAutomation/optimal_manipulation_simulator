from meshpy import VirtualCamera, ObjFile, SceneObject
from perception import CameraIntrinsics
from visualization import Visualizer2D
from core import RigidTransform
import numpy as np
from matplotlib import pyplot as plt
from perception.object_render import RenderMode

dexnet_path = "/home/chris/optimal_manipulation_simulator/dex-net-new-api"

if __name__ == "__main__":
    print "\n\n\n\n"
    camera_intr = CameraIntrinsics.load("../config/primesense_overhead.intr")
    camera = VirtualCamera(camera_intr)

    of = ObjFile(dexnet_path + "/data/meshes/chess_pieces/WizRook.obj")
    rook1 = of.read()


    T_world_camera = RigidTransform(rotation = np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
                                    translation=np.array([0,0,.2]).T,
                                    from_frame="world",
                                    to_frame="primesense_overhead")
    # T_world_camera = RigidTransform(rotation = np.array([[1,0,0],[0,1,0],[0,0,1]]),
    #                                 translation=np.array([-.2,0,0]).T,
    #                                 from_frame="world",
    #                                 to_frame="primesense_overhead")

    # squares = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    # for i, square in enumerate(squares):

    T_rook2_world = RigidTransform(rotation = np.eye(3),
                                   translation = np.array([.06, .06, 0]).T,
                                   from_frame="rook2",
                                   to_frame="obj")
    of2 = ObjFile(dexnet_path + "/data/meshes/chess_pieces/WizRook.obj")
    rook2 = of2.read()
    camera.add_to_scene("rook2", SceneObject(rook2, T_rook2_world))

    T_table_world = RigidTransform(rotation = np.eye(3),
                                   translation = np.array([0, 0, -.0025]).T,
                                   from_frame="table",
                                   to_frame="obj")
    of3 = ObjFile(dexnet_path + "/data/meshes/Table_triangulated_faces_resized.obj")
    table = of3.read()
    camera.add_to_scene("table", SceneObject(table, T_table_world))

    of3 = ObjFile(dexnet_path + "/data/meshes/chess_pieces/WizRook.obj")
    rook3 = of3.read()
    T_rook3_world = RigidTransform(rotation = np.eye(3),
                                   translation = np.array([-.06, .06, 0]).T,
                                   from_frame="rook3",
                                   to_frame="obj")
    camera.add_to_scene("rook3", SceneObject(rook3, T_rook3_world))

    # depth_im = camera.wrapped_images(table, [T_world_camera], RenderMode.DEPTH_SCENE)
    depth_im = camera.wrapped_images(rook1, [T_world_camera], RenderMode.DEPTH_SCENE)
    Visualizer2D.imshow(depth_im[0].image)
    Visualizer2D.show()