from .environment import Environment
from os import path
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
from typing import Optional
from math import pi

import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

"""OpenAI Gym Pendulum-v0 environment."""
class Pendulum(Environment):
    """Creates a new Pendulum environment."""
    def __init__(self):
        """Creates a new Pendulum environment.
        
           EXAMPLE
               >>> env = Pendulum()
               >>> print(env.states)
               3
               >>> print(env.actions)
               1
        """
        self.env = gym.make("Pendulum-v1")        
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.observation_space.shape[0]

    def step(self, u):
        return self.env.step(np.atleast_1d(u))

    def plotnetwork(self, network):
        """Plot value function and policy.
        
           plot(ddpg) plots the value function and policy of network `ddpg`.
        """
        xx, yy = np.meshgrid(np.linspace(-np.pi,np.pi, 64), np.linspace(-8, 8, 64))
        obs = np.hstack((np.reshape(np.cos(xx), (xx.size, 1)),
                         np.reshape(np.sin(xx), (xx.size, 1)),
                         np.reshape(       yy , (xx.size, 1))))
        cc = np.reshape(network.critic(obs), xx.shape)
        aa = np.reshape(network.actor(obs), xx.shape)

        fig, axs = plt.subplots(1,2)
        fig.subplots_adjust(right=1.2)

        h = axs[0].contourf(xx, yy, cc, 256)
        fig.colorbar(h, ax=axs[0])
        h = axs[1].contourf(xx, yy, aa, 256)
        fig.colorbar(h, ax=axs[1])
        
        axs[0].set_title('Critic')
        axs[1].set_title('Actor')
        
    def plotlinear(self, w, theta, feature=None):
        """Plot value function and policy.
        
           plot(w, feature) plots the function approximated by 
           w^T feature(x) .
           
           plot(w, theta, feature) plots the functions approximated by 
           w^T * feature(x) and theta^T * feature(x) .
        """
        ac = True
        if feature is None:
            feature = theta
            ac = False
        
        p, v = np.meshgrid(np.linspace(-pi, pi, 64), np.linspace(-8, 8, 64))
        s = np.vstack((np.cos(p.flatten()), np.sin(p.flatten()), v.flatten())).T
        f = feature(s)
        c = np.reshape(np.dot(f, w), p.shape)
        
        if ac:
            a = np.reshape(np.dot(f, theta), p.shape)
        
            fig, axs = plt.subplots(1,2)
            fig.subplots_adjust(right=1.2)

            h = axs[0].contourf(p, v, c, 256)
            fig.colorbar(h, ax=axs[0])

            h = axs[1].contourf(p, v, a, 256)
            fig.colorbar(h, ax=axs[1])

            axs[0].set_title('Critic')
            axs[1].set_title('Actor')
        else:
            fig, ax = plt.subplots(1,1)
            h = ax.contourf(p, v, c, 256)
            fig.colorbar(h, ax=ax)
            
            ax.set_title('Approximator')
            