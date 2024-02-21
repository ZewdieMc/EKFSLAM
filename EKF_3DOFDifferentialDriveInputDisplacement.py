from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        xk_1 = xk_1.reshape(3, 1)  # Reshape xk_1 to have shape (3, 1)
        uk = uk.reshape(3, 1)  # Reshape uk to have shape (3, 1)
        xk_bar = Pose3D(xk_1).oplus(uk)

        return xk_bar

    def Jfx(self, xk_1):
        # TODO: To be completed by the student
        xk_1 = xk_1.reshape(3, 1)  # Reshape xk_1 to have shape (3, 1)
        J = Pose3D(xk_1).J_1oplus(self.uk)

        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        xk_1 = xk_1.reshape(3, 1)  # Reshape xk_1 to have shape (3, 1)
        J = Pose3D(xk_1).J_2oplus()

        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        h = xk[2,0]
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
        # TODO: To be completed by the student
        uk_pulse , Qk = self.robot.ReadEncoders()
        # Compute travel distance of 2 wheels [meter] from output of the encoder
        d_L   =  uk_pulse[0, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)
        d_R   =  uk_pulse[1, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)

        # Compute travel distance of the center point of robot between k-1 and k
        d     =  (d_L + d_R) / 2.
        # Compute rotated angle of robot around the center point between k-1 and k
        delta_theta_k  =  np.arctan2(d_R - d_L, self.wheelBase)

        # Compute xk from xk_1 and the travel distance and rotated angle. Got the equations from chapter 1.4.1: Odometry 
        uk             = np.array([[d],
                                    [0],
                                    [delta_theta_k]])
        Qk = np.diag(np.array([0.04 ** 2, 0.02 ** 2, np.deg2rad(0.5) ** 2]))  # covariance of simulated displacement noise

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student
        # Read compass sensor
        zk, Rk  = self.robot.ReadCompass()

        # Compute H matrix
        Hk      = np.array([0., 0., 1.]).reshape((1,3))
        # Compute V matrix
        Vk      = np.diag([1.])
        # Raise flag got measurement
        if len(zk) != 0:
            self.headingData = True

        return zk, Rk, Hk, Vk


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)