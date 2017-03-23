from meshpy import VirtualCamera, ObjFile, SceneObject
from perception import CameraIntrinsics
from core import RigidTransform
import numpy as np
dexnet_path = "/home/cloudminds/chomp_ws/src/dex-net-new-api"

if __name__ == "__main__":
    camera_intr = CameraIntrinsics("world", fx=1000, height=480, width=360)
    camera = VirtualCamera(camera_intr)
    # camera = perception.VirtualKinect2Sensor()
    of = ObjFile(dexnet_path + "/data/meshes/chess_pieces/WizRook.obj")
    mesh = of.read()
    # position = RigidTransform(from_frame="rook")
    # scene_rook = SceneObject(mesh, position)
    T_world_camera = RigidTransform(rotation = np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
                                    translation=np.array([0,0,1]).T,
                                    from_frame="world",
                                    to_frame="camera")
    camera.images(mesh, [T_world_camera]);