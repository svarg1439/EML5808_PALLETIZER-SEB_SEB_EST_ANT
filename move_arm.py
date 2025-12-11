
import numpy as np
import time
import rclpy
from math import sqrt, sin, cos
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from bottles_demo_node import DetectAndPickNode


# Move Js are forward kinematics, use the suffix _IK for inverse kinematics.
# Move Ls are the opposite, use the suffix "_FK" for forward kinematics.
# Moves take in a position vector, then velocity, and acceleration



class EML5808_Palletizer:
    def __init__(self, ip):
        
        self.jointnames = ["base", "shldr", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.coordinatenames = ["X", "Y", "Z", "ROLL", "PITCH", "YAW"]
        self.ROBOT_IP = ip                                                              # UR20
        self.vel = 0.2                                                                  # IN M/S
        self.fast_vel = 0.7
        self.accel = 0.5                                                                # IN M/S^2
        self.force_thresh = [0.1, 0.1, 1.1, 0.1, 0.1, 0.1]
        self.vel_vector = [0.0, 0.0, -0.1, 0.0, 0.0, 0.0]
        self.tool_offset = 0.040                                                        # IN M
        self.err_x = -0.000
        self.err_y = 0.086
        self.err_z = 0.173
        self.home_pose = [self.deg2rad(0.0), self.deg2rad(-90.0), self.deg2rad(-30.0), self.deg2rad(-150.0), self.deg2rad(90.0), self.deg2rad(180.0)]              # Joint state angles in radians 
        
        self.rtde_c = RTDEControlInterface(self.ROBOT_IP)
        self.rtde_r = RTDEReceiveInterface(self.ROBOT_IP)
        print("Connecting to robot @", self.ROBOT_IP)
        
    def rad2deg(self, rad_angle):
        deg_angle = rad_angle*180/3.14159
        return deg_angle
    
    def deg2rad(self, deg_angle):
        rad_angle = deg_angle*3.14159/180
        return rad_angle
        
    def print_tcp_pose(self):
        da_pose = self.rtde_r.getActualTCPPose()
        i=0
        print("[INFO] CURRENT TCP POSE:")
        for n in da_pose:
            print(self.coordinatenames[i], "\t:\t", f"{n:.1f}")
            i=i+1
        print('\n')
    
    def go_home(self):
        self.rtde_c.moveJ(self.home_pose, self.fast_vel, self.accel)
        
    def test_box_coordinates(self):
        box1 = [1.0, 0.0, 0.2, 2.2, 2.2, 0.0]                                   # XYZ and RPY in meters and radians respectively
        box2 = [0.8, 0.3, 0.2, 2.2, 2.2, 0.0]  
        box3 = [1.4, -0.3, 0.2, 2.2, 2.2, 0.0]                                  # Ditto ^
        box4 = [1.0, 0.0, 0.2, 2.2, 2.2, 0.0]                                   # XYZ and RPY in meters and radians respectively
        box5 = [0.8, 0.3, 0.2, 2.2, 2.2, 0.0]  
        box6 = [1.4, -0.3, 0.2, 2.2, 2.2, 0.0]                                  # Ditto ^
        combined_box = box1+box2+box3+box4+box5+box6                            # REPLACE COORDINATES WITH ACTUAL BOX FINDING FUNCTION USING YOLO

        # Expecting algorithm to export a large list containing
        # all box coordinates. Below is a function to extract 
        # the locations and place them in a list of lists for 
        # easy information extraction. 

        boxes = []                                                              
        box_count = len(combined_box)//6
        for i in range(box_count):
            start = i*6
            end = start+6
            boxes.append(combined_box[start:end])
        return boxes
    
    def pallet_grid(self, position):
        grid =        [[-0.5, -0.800, 0.250, 2.2, 2.2, 0.0],
                       [-0.5, -1.400, 0.250, 2.2, 2.2, 0.0],
                       [0.5, -1.400, 0.250, 2.2, 2.2, 0.0],
                       [0.5, -0.850, 0.250, 2.2, 2.2, 0.0]]
        return grid[position]
    
    def palletize(self, box_coordinates, box_num, grid_num):
        box = self.transform_camera_to_base(box_coordinates[box_num])
        box_hover = self.transform_camera_to_base(box_coordinates[box_num])
        box[0] = box[0]+self.err_x
        box[1] = box[1]+self.err_y
        box[2] = box[2]+self.err_z
        box_hover[0] = box_hover[0]+self.err_x
        box_hover[1] = box_hover[1]+self.err_y
        box_hover[2] = 0.200
        box_pickup_home = [self.deg2rad(0.0), self.deg2rad(-60.0), self.deg2rad(-120.0), self.deg2rad(-90.0), self.deg2rad(90.0), self.deg2rad(180.0)]
        
        print("[INFO] BOX COORDINATES RELATIVE TO UR BASE: ", box)
        
        print("\n[INFO] MOVING TO BOX HOVER POINT", box_num+1)
        time.sleep(.5)
        self.rtde_c.moveJ_IK(box_hover, self.fast_vel, self.accel)
        self.print_tcp_pose()
        time.sleep(1.0)

        print("[INFO] PICKING UP BOX")
        # self.rtde_c.moveL(box, self.vel, self.accel)
        self.rtde_c.moveUntilContact(self.vel_vector)
        self.print_tcp_pose()
        self.rtde_c.moveL(box_hover, self.vel, self.accel)
        self.rtde_c.moveJ(box_pickup_home, self.vel, self.accel)
        
        # MOVE TO PALLET
        pallet_home = [self.deg2rad(-90.0), self.deg2rad(-60.0), self.deg2rad(-90.0), self.deg2rad(-120.0), self.deg2rad(90.0), self.deg2rad(180.0)]
        pallet_drop = self.pallet_grid(grid_num)
        pallet_drop[2] = box[2]

        print("[INFO] MOVING TO PALLET")
        time.sleep(.5)
        self.rtde_c.moveJ(pallet_home, self.fast_vel, self.accel)

        print("[INFO] MOVING TO PALLET HOVER POINT")
        self.rtde_c.moveJ_IK(self.pallet_grid(grid_num), self.fast_vel, self.accel)

        print("[INFO] DROPPING OFF BOX")
        # self.rtde_c.moveL(pallet_drop, self.vel, self.accel)
        self.rtde_c.moveUntilContact(self.vel_vector)
        time.sleep(.5)
        self.rtde_c.moveL(self.pallet_grid(grid_num), self.vel, self.accel)
        
        # Set up for next pickup
        print("[INFO] GOING TO HOME")
        self.rtde_c.moveJ(pallet_home, self.fast_vel, self.accel)
        # self.rtde_c.moveJ(box_pickup_home, self.fast_vel, self.accel)

    def transform_camera_to_base(self, box_coordinates):
        """
        Converts box XYZ from camera frame into UR base frame.
        Returns [x_base, y_base, z_base, roll, pitch, yaw]
        where r,p,y are taken directly from the input list.
        """
        # --------------------------------------------------------
        # 1. Convert UR axis-angle (rx, ry, rz) to rotation matrix
        # --------------------------------------------------------
        def axis_angle_to_rot(rx, ry, rz):
            theta = sqrt(rx*rx + ry*ry + rz*rz)
            if theta < 1e-9:
                return np.eye(3)

            kx, ky, kz = rx/theta, ry/theta, rz/theta
            c = cos(theta)
            s = sin(theta)
            v = 1 - c

            R = np.array([
                [kx*kx*v + c,     kx*ky*v - kz*s, kx*kz*v + ky*s],
                [ky*kx*v + kz*s,  ky*ky*v + c,    ky*kz*v - kx*s],
                [kz*kx*v - ky*s,  kz*ky*v + kx*s, kz*kz*v + c]
            ])
            return R

        # --------------------------------------------------------
        # 2. Camera → End Effector transform (fixed)
        # --------------------------------------------------------
        R_ee_cam = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ])

        t_ee_cam = np.array([0.0, 0.06, 0.06])   # meters

        T_ee_cam = np.eye(4)
        T_ee_cam[:3, :3] = R_ee_cam
        T_ee_cam[:3, 3]  = t_ee_cam

        # --------------------------------------------------------
        # 3. End Effector → Base transform from UR robot
        # --------------------------------------------------------
        tcp = self.rtde_r.getActualTCPPose()  # [x,y,z,rx,ry,rz]
        x, y, z, rx, ry, rz = tcp

        R_base_ee = axis_angle_to_rot(rx, ry, rz)

        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R_base_ee
        T_base_ee[:3, 3]  = [x, y, z]

        # --------------------------------------------------------
        # 4. Extract XYZ and RPY from input box coordinates
        # --------------------------------------------------------
        bx, by, bz, roll, pitch, yaw = box_coordinates

        p_cam = np.array([bx, by, bz, 1.0])  # homogeneous position

        # --------------------------------------------------------
        # 5. Compute Base←EE←Cam transform
        # --------------------------------------------------------
        p_base = T_base_ee @ T_ee_cam @ p_cam
        x_b, y_b, z_b = p_base[:3]

        # --------------------------------------------------------
        # 6. Return result with original RPY
        # --------------------------------------------------------
        
        box_rel_2_base = [float(x_b), float(y_b), float(z_b), roll, pitch, yaw]
        return box_rel_2_base

    def shutdown(self):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

def look4boxes(args=None):
    global box_coor
    box_coor = []
    rclpy.init(args=args)
    node = DetectAndPickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[INFO] Node interrupted.")
    finally:
        box_coor = node.get_box_coor()
        if not box_coor:
            print("[ERROR] DID NOT RECIEVE BOX COORDINATES")
        else:
            print("[INFO] RECEIVED COORDINATES FOR BOXES")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


#------------------------------------MAIN_CODE-----------------------------------------------


if __name__ == "__main__":
    initialize = EML5808_Palletizer("172.16.3.14")
    print("[INFO] GOING TO HOME POSE")
    initialize.go_home()
    initialize.shutdown()

    print("[INFO] LOOKING FOR BOXES")
    look4boxes()

    ROBOT_IP = "172.16.3.14"                # UR20
    #ROBOT_IP = "192.168.8.161")             # UR3
    print("[INFO] CONNECTING TO ROBOT AT ", ROBOT_IP)
    robot = EML5808_Palletizer(ROBOT_IP)
    
    print("[INFO] INITIATING 3 SECOND DELAY UNTIL START")
    print("[INFO] ENSURE EVERYONE YOU LIKE IS OUTSIDE THE RANGE OF THE ROBOT")
    time.sleep(3)
    print("[INFO] STARTING PALLETIZING MOVEMENT\n")
    time.sleep(1)

    print("[INFO] USING BOX COORDINATES: ", box_coor)
    # boxes = robot.test_box_coordinates()
    boxes = box_coor
    
    i = 0
    j = 0
    for n in boxes:
        if j > 3:
            j = 0
        print("[INFO] PALLETIZING BOX ", i, "AT GRID POINT ", j)
        robot.palletize(boxes, i, j)
        i = i+1
        j = j+1
  
    robot.shutdown()
    print("[INFO] DISCONNECTED")