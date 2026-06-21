from controller import Controller
from datatypes import ControllerConfig, PlantConfig
import numpy as np
from pid import PIDController

def check_linerised_position(s_clean:np.ndarray)-> bool:
    if abs(np.pi - s_clean[1]) < np.radians(50):
        return True
    return False

class SwingUp(Controller):
    def __init__(self,config:ControllerConfig, K:float, plant_config:PlantConfig) -> None:
        super().__init__(config)
        self.name = "SwingUp"
        self._K = K
        self._plant_config = plant_config.copy()
    
    def get_action(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        E = 0.5*self._plant_config.m1*self._plant_config.L1*(s_clean[4])**2 \
             - self._plant_config.m1*self._plant_config.g*self._plant_config.L1*(1-np.cos(s_clean[1]))
        E_t = -2*self._plant_config.m1*self._plant_config.g*self._plant_config.L1
        F = self._K*(E-E_t)*np.sign(s_clean[4]*np.cos(s_clean[1]))
        return F
class SwingUpAndBalance(Controller):
    def __init__(self,config:ControllerConfig, swingup_controller:SwingUp, balance_controller:PIDController):
        super().__init__(config)
        self.name = "BEAST"
        self._swing_up_controller = swingup_controller
        self._balance_controller = balance_controller

    def get_action(self, s_clean: np.ndarray, target_state: np.ndarray) -> float:
        if check_linerised_position(s_clean):
            return self._balance_controller.get_action(s_clean, target_state)
        return self._swing_up_controller.get_action(s_clean, target_state)