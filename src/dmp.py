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

class GaussianBasis:
    hi: NDArray[ParameterType]
    ci: NDArray[ParameterType]

    def __init__(self, hi: NDArray[ParameterType], ci: NDArray[ParameterType]):
        if ci.shape != hi.shape and ci.ndim == 1:
            raise ValueError("NDArrays ci and hi must have the same shape and can only be one-dimensional ndarrays.")

        self.hi = hi
        self.ci = ci

    def eval_at(self, x: ParameterType):
        return np.exp(-1*self.hi * (x - self.ci))

    def get_size(self):
        return self.ci.shape[0]



class DMP1Dim:
    """
    Realizes a dmp system. It requires:
     - parameters ci and hi defining a set of basis functions for the learned force function.
     - system parameters of type DMPParam
     - optionally: a user-defined initial system state
    """
    basis: GaussianBasis
    params: DMPParam
    wi: NDArray[ParameterType]

    initial_system_state: StateVectorType
    system_state: StateVectorType

    def __init__(self, basis: GaussianBasis, parameters: DMPParam, weights: NDArray[ParameterType] | None = None, initial_state: StateVectorType | None = None):
        self.basis = basis
        self.params = parameters

        if initial_state is None:
            initial_state = np.array([0,0,1], ParameterType)

        if weights is None:
            weights = np.random.rand(basis.get_size())

        if initial_state.shape != (3,):
            raise ValueError(f"Expected shape of initial_state to be (3,), got {initial_state.shape} instead.")

        if weights.shape != (basis.get_size(),):
            raise ValueError(f"Expected shape of weights to be ({basis.get_size()}, ), got {weights.shape} instead.")

        self.initial_system_state = initial_state.copy()
        self.system_state = self.initial_system_state.copy()
        self.wi = weights



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
        psi = self.basis.eval_at(self.system_state[2])
        return np.dot(self.wi, psi)/np.sum(psi)


    def execute(self, delta_t: ParameterType) -> StateVectorType:
        """
        :param delta_t: the delta in time that has passed since the previous execution step
        :return: the current system state
        """
        dz = self.get_z_dot()*delta_t
        self.update_system_state(dz)
        return self.system_state