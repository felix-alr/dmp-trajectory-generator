import numpy as np


class DMPParam:
    """
    Defines the data type DMPParam and holds all the required parameters of the dmp system.
    """
    alpha_y: float
    beta_y: float
    alpha_x: float
    tao: float

    def __init__(self, alpha_y: float, beta_y: float, alpha_x: float, tao: float):
        self.alpha_y = alpha_y
        self. beta_y = beta_y
        self.alpha_x = alpha_x
        self.tao = tao


class DMP1Dim:
    """
    Realizes a dmp system. It requires:
     - parameters ci and hi defining a set of basis functions for the learned force function.
     - system parameters of type DMPParam
     - optionally: a user-defined initial system state
    """
    ci: np.ndarray
    hi: np.ndarray
    params: DMPParam

    initial_system_state: np.ndarray
    system_state: np.ndarray

    def __init__(self, ci: np.ndarray, hi: np.ndarray, parameters: DMPParam, initial_state: np.ndarray = np.array([0,0,1])):
        self.ci = ci
        self.hi = hi
        self.params = parameters

        self.initial_system_state = initial_state
        self.system_state = self.initial_system_state



    def update_system_state(self, dz: np.ndarray):
        """
        :param dz: the difference
        :return:
        """
        self.system_state += dz


    def reset_system_state(self):
        """
        This function resets the dmp system and thus the trajectory.
        """
        self.system_state = self.initial_system_state


    def get_z1_dot(self) -> float:
        """
        :return: temporal derivative of the z1 state variable
        """
        return self.system_state[1]


    def get_z2_dot(self) -> float:
        """
        :return: temporal derivative of the z2 state variable
        """
        z = self.system_state
        return (-self.params.alpha_y*self.params.beta_y/self.params.tao) * z[0]\
            - (self.params.alpha_y/self.params.tao) * z[1]\
            + self.f()/self.params.tao


    def get_z3_dot(self) -> float:
        """
        :return: temporal derivative of the z3 state variable
        """
        return (-self.params.alpha_x/self.params.tao) * self.system_state[2]


    def get_z_dot(self) -> np.ndarray:
        """
        :return: ndarray of the system dynamics
        """
        return np.array(self.get_z1_dot(), self.get_z2_dot(), self.get_z3_dot())


    def f(self) -> float:
        """
        :return: the learned force function output.
        """
        return 0


    def execute(self, delta_t: float) -> np.ndarray:
        """
        :param delta_t: the delta in time that has passed since the previous execution step
        :return: the current system state
        """
        dz = self.get_z_dot()*delta_t
        self.update_system_state(dz)
        return self.system_state