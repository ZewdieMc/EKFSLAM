from Localization import *
from Pose import *
import numpy as np

class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

        # Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        self.uk = uk  # store input
        # TODO: to be completed by the student
        dl = uk[0][0] * 2 * np.pi * self.wheelRadius / self.robot.pulse_x_wheelTurns
        dr = uk[1][0] * 2 * np.pi * self.wheelRadius / self.robot.pulse_x_wheelTurns

        vl = dl/self.dt
        vr = dr/self.dt


        vx = (vr + vl) / 2
        w = (vr - vl)/ self.wheelBase# proof this

        v = np.array([[vx],[0], [w]])

        displacment  = v * self.dt


        #d = dr + dl)/2
        # theta = xk_1[2, 0] + np.arctan2((dr - dl), self.wheelBase) 
        # xk = (np.array([[xk_1[0, 0] + d * np.cos(theta)],
        #                       [xk_1[1, 0] + d * np.sin(theta)],
        #                       [theta]]))
        xk = xk_1.oplus(displacment)
        return xk

    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """
        rsk, zsk = self.robot.ReadEncoders()
        # where am i gonna use it?zsk
        return rsk
        # TODO: to be completed by the student

        pass

