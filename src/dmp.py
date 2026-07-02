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
    g: StateVectorType

    def __init__(self, alpha_y: ParameterType, beta_y: ParameterType, alpha_x: ParameterType, tao: ParameterType, g: StateVectorType):
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
        return np.exp(-1*self.hi[:, None] * (x[None, :] - self.ci[:, None]))

    def get_size(self):
        return self.ci.shape[0]



class DMPNDim:
    """
    Realizes a dmp system. It requires:
     - GaussianBasis defining a set of basis functions for the learned force function.
     - the corresponding weights for the learned force function.
     - system parameters of type DMPParam.
     - optionally: a user-defined initial system state
    """
    params: DMPParam
    basis: GaussianBasis
    wi: NDArray[ParameterType] # Shape: (dim, basis.get_size())

    initial_system_state: StateVectorType # Shape: (dim, 3)
    system_state: StateVectorType         # Shape: (dim, 3)

    dim: int

    def __init__(self, parameters: DMPParam, basis: GaussianBasis, weights: NDArray[ParameterType] | None = None, initial_state: StateVectorType | None = None, dim: int = 3):
        """
        :param basis: a GaussianBasis object defining all the basis functions used for the force function.
        :param parameters: a DMPParam object containing all relevant parameters for the dmp.
        :param weights: the weights for each of the Gaussian bases for constructing the force function.
        :param initial_state: the initial state of the dmp system. Shape: (n, 3)
        """
        self.basis = basis
        self.params = parameters

        if initial_state is None:
            initial_state = np.array([[0, 0, 1]] * dim, ParameterType)

        if weights is None:
            weights = np.random.rand(dim, basis.get_size())

        if initial_state.ndim != 2 or initial_state.shape[0] != dim or initial_state.shape[1] != 3:
            raise ValueError(f"Expected shape of initial_state to be ({dim}, 3), got {initial_state.shape} instead.")

        if weights.ndim != 2 or weights.shape[0] != dim or weights.shape[1] != basis.get_size():
            raise ValueError(f"Expected shape of weights to be ({dim}, {basis.get_size()}), got {weights.shape} instead.")

        self.initial_system_state = initial_state.copy()
        self.system_state = self.initial_system_state.copy()
        self.wi = weights.copy()



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


    def get_z1_dot(self) -> StateVectorType:
        """
        :return: temporal derivative of the z1 state variable
        """
        return self.system_state[:, 1]


    def get_z2_dot(self) -> StateVectorType:
        """
        :return: temporal derivative of the z2 state variable
        """
        z = self.system_state
        return (-self.params.alpha_y*self.params.beta_y/pow(self.params.tao, 2)) * z[:, 0]\
            - (self.params.alpha_y/self.params.tao) * z[:, 1]\
            + (self.params.alpha_y*self.params.beta_y/pow(self.params.tao, 2))*self.params.g\
            + np.dot(z[:, 2], self.f())/pow(self.params.tao, 2)


    def get_z3_dot(self) -> StateVectorType:
        """
        :return: temporal derivative of the z3 state variable
        """
        return (-self.params.alpha_x/self.params.tao) * self.system_state[:, 2]


    def get_z_dot(self) -> StateVectorType:
        """
        :return: ndarray of the system dynamics
        """
        return np.array([self.get_z1_dot(), self.get_z2_dot(), self.get_z3_dot()], dtype=ParameterType)


    def f(self) -> StateVectorType:
        """
        :return: the learned force function output.
        """
        psi = self.basis.eval_at(self.system_state[:, 2])
        return np.matmul(self.wi, psi)/np.sum(psi)


    def execute(self, delta_t: ParameterType) -> StateVectorType:
        """
        :param delta_t: the delta in time that has passed since the previous execution step
        :return: the current system state
        """
        dz = np.transpose(self.get_z_dot())*delta_t
        self.update_system_state(dz)
        return self.system_state