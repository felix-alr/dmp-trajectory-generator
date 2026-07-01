import numpy as np

from numpy.typing import (NDArray)
from dataclasses import dataclass

ParameterType = np.float32
StateVectorType = NDArray[ParameterType]


@dataclass
class DMPParam:
    """
    Defines the data type DMPParam and holds all the required parameters of the dmp system.
    """
    alpha_y: ParameterType
    beta_y: ParameterType
    alpha_x: ParameterType
    tao: ParameterType
    g: ParameterType

    def __init__(self, alpha_y: ParameterType, beta_y: ParameterType, alpha_x: ParameterType, tao: ParameterType, g: ParameterType):
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.alpha_x = alpha_x
        self.tao = tao
        self.g = g


class DMP1Dim:
    """
    Realizes a dmp system. It requires:
     - parameters ci and hi defining a set of basis functions for the learned force function.
     - system parameters of type DMPParam
     - optionally: a user-defined initial system state
    """
    ci: NDArray[ParameterType]
    hi: NDArray[ParameterType]
    params: DMPParam

    initial_system_state: NDArray[ParameterType]
    system_state: NDArray[ParameterType]

    def __init__(self, ci: NDArray[ParameterType], hi: NDArray[ParameterType], parameters: DMPParam, initial_state: StateVectorType | None = None):
        self.ci = ci
        self.hi = hi
        self.params = parameters

        if initial_state is None:
            initial_state = np.array([0,0,1], ParameterType)

        if initial_state.shape != (3,):
            raise ValueError(f"Expected shape of initial_state to be (3,), got {initial_state.shape}")

        self.initial_system_state = initial_state.copy()
        self.system_state = self.initial_system_state.copy()



    def update_system_state(self, dz: StateVectorType):
        """
        :param dz: the difference
        :return:
        """
        self.system_state += dz


    def reset_system_state(self):
        """
        This function resets the dmp system and thus the trajectory.
        """
        self.system_state = self.initial_system_state.copy()


    def get_z1_dot(self) -> ParameterType:
        """
        :return: temporal derivative of the z1 state variable
        """
        return self.system_state[1]


    def get_z2_dot(self) -> ParameterType:
        """
        :return: temporal derivative of the z2 state variable
        """
        z = self.system_state
        return (-self.params.alpha_y*self.params.beta_y/pow(self.params.tao, 2)) * z[0]\
            - (self.params.alpha_y/self.params.tao) * z[1]\
            + (self.params.alpha_y*self.params.beta_y/pow(self.params.tao, 2))*self.params.g\
            + z[2]*self.f()/pow(self.params.tao, 2)


    def get_z3_dot(self) -> ParameterType:
        """
        :return: temporal derivative of the z3 state variable
        """
        return (-self.params.alpha_x/self.params.tao) * self.system_state[2]


    def get_z_dot(self) -> StateVectorType:
        """
        :return: ndarray of the system dynamics
        """
        return np.array([self.get_z1_dot(), self.get_z2_dot(), self.get_z3_dot()], dtype=ParameterType)


    def f(self) -> ParameterType:
        """
        :return: the learned force function output.
        """
        return 0


    def execute(self, delta_t: ParameterType) -> StateVectorType:
        """
        :param delta_t: the delta in time that has passed since the previous execution step
        :return: the current system state
        """
        dz = self.get_z_dot()*delta_t
        self.update_system_state(dz)
        return self.system_state