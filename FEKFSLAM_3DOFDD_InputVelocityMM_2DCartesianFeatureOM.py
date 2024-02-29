from FEKFSLAM import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from Pose import *
from blockarray import *
from MapFeature import *
import numpy as np
from FEKFSLAMFeature import *

class FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    def __init__(self, *args):

        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
       
        super().__init__(*args)


    # def GetFeatures(self):
    # Get features is inherited from EKF_3DOFDifferentialDriveInputDisplacement


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    kSteps = 5000
    alpha = 0.99

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))
    dr_robot = DR_3DOFDifferentialDrive(index, kSteps, robot, x0)
    robot.SetMap(M)

    auv = FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM([], alpha, kSteps, robot)

    P0 = np.zeros((3, 3))
    usk=np.array([[0.5, 0, 0.03]]).T
    print("x0: ", x0)
    print("P0: ", P0)
    f1 = M[0]
    f2 = M[1]
    print("f1: ", f1.shape)
    x0 = np.block([[x0], [f1]])
    print("x0: ", x0)
    # P0_hstack = [P0, )]
    P0_left = np.zeros((P0.shape[0], f1.shape[0]))
    P0_hstack = np.block([P0, P0_left])

    P0_bottom = np.zeros((f1.shape[0], P0.shape[1]))
    P0_bottom = np.block([P0_bottom, np.zeros((f1.shape[0], f1.shape[0]))])

    print("P0_hstack: ", P0_hstack.shape)
    print("P0_bottom: ", P0_bottom.shape)
    P0 = np.block([[P0_hstack], [P0_bottom]])
    print("________________")
    print("P0: ", P0)
    print("P0.shape:", P0.shape)
    
    # auv.LocalizationLoop(x0, P0, usk)

    exit(0)
