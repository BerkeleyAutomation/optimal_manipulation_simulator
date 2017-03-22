import meshpy
import perception
dexnet_path = "/home/cloudminds/chomp_ws/src/dex-net-new-api"

if __name__ == "__main__":
    # camera_intr = CameraIntrinsics("world", fx=f, fy=f,
    #                                cx=cx, cy=cy, skew=0.0,
    #                                height=self.im_height, width=self.im_width)
    # camera = meshpy.VirtualCamera(camera_intr)
    camera = perception.VirtualKinect2Sensor()
    of = meshpy.ObjFile(dexnet_path + "/data/meshes/chess_pieces/WizRook.obj")
    mesh = of.read()
    