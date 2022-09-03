
from os import path
import numpy as np
import gym
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
from gym import spaces
from typing import Optional

class Tweezer(gym.Env):
    """
       ### Description
        To be implemented: insert here description about the state variables
    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
    Action Space
    
    Observation Space
    Rewards
    Starting State
    Episode Termination
    Arguments
    Version History
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, pressure=10**1, render_mode: Optional[str] = None):
        avogrado = 6.02*(10**23) # 1/mol
        kB = 1.380649*(10**-23) # m^2 kg /(s^2 K)
        c = 3*(10**8) # m/s
        self.m_gas_molecule = 0.02897/avogrado  # kg
        self.max_voltage = 100 # V
        self.m = 1.14*(10**-18) # kg
        self.d = 11*(10**-3) # m
        self.R = 50*(10**-9) # m
        self.T = 273.5 + 25 # K
        self.pressure = pressure # Pa
        self.Q = 2*(10**4)*(1.6*(10**-19))*np.power(self.R/(2.5*10**-6), 2) # Coulomb
        self.gas_velocity = np.sqrt((3*kB*self.T/self.m_gas_molecule)) # m/s
        self.omega_0x = 2*np.pi*150*(10**3) # rad/s
        self.omega_0y = 2*np.pi*150*(10**3) # rad/s
        self.gamma = 15.8*(np.power(self.R, 2)*self.pressure)/(self.gas_velocity*self.m) # kg/s
        self.dt = 2*10**-8 # s
        self.noise_amplitude = np.sqrt(2*kB*self.T*self.m*self.gamma) # N
        self.std_noise = 100# adimensional
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([10**-3, 10**-3, 1, 1, 100, 100], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(shape=(6,), 
                                            low=-high, 
                                            high=high, 
                                            dtype=np.float32)

    def white_noise(self):
        return self.noise_amplitude*np.random.normal(0, self.std_noise)

    def step(self, u):
        terminal = False
        x, y, xdot, ydot, xddot, yddot = self.state  # th := theta
        gamma = self.gamma
        omega_0x = self.omega_0x
        omega_0y = self.omega_0y
        m = self.m
        Q = self.Q 
        d = self.d 
        dt = self.dt
        u = np.clip(u, -self.max_voltage, self.max_voltage)
        eletric_F = Q*u/d
        self.last_u = u  # for rendering
        x_i_plus = x + xdot*dt+0.5*dt*dt*xddot
        xddot_i_plus = (1/m)*(eletric_F + self.white_noise()) - np.power(omega_0x, 2)*x_i_plus - gamma*xdot
        xdot_i_plus = xdot + 0.5*(xddot_i_plus+xddot)*dt
        y_i_plus = y + ydot*dt+0.5*dt*dt*yddot
        yddot_i_plus = (1/m)*(eletric_F + self.white_noise()) - np.power(omega_0y, 2)*y_i_plus - gamma*ydot
        ydot_i_plus = ydot + 0.5*(yddot_i_plus + yddot)*dt
    
        self.state = np.array([x_i_plus, 
                               y_i_plus, 
                               xdot_i_plus, 
                               ydot_i_plus, 
                               xddot_i_plus,
                               yddot_i_plus])
        #costs = (x_i_plus/10**-7)**2+(v_i_three_half/0.1)**2
#        self.renderer.render_step()
        costs = 0
#        if abs(x_i_plus) > 10**-7 or abs(v_i_three_half) > 0.1:
#            terminal = True
#            costs = abs(x_i_plus+v_i_three_half)
        return self._get_obs(), -costs, terminal, u, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        high = np.array([20*10**-8, 20*10**-8, 0.0001, 0.0001, 0, 0])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.renderer.reset()
        self.renderer.render_step()
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def _get_obs(self):
        x, y, xdot, ydot, xddot, yddot = self.state
        return np.array([y, x, xdot, ydot, xddot, yddot], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def _render(self, mode="human"):
         pass

    def close(self):
        pass