"""ELE2761 Exercise 4: Policy approximation.

Implements the provided functionality to be used in your solution.

CLASSES
    DDPG         -- Deep Deterministic Policy Gradient network
    Memory       -- Replay Memory

FUNCTIONS
    rbfprojector -- Gaussian RBF projector factory.
"""

from math import pi
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

def __gaussrbf(s, p, v, sigma):
    """Gaussian radial basis function activation.
    
       f = gaussrbf(s, p, v, sigma) returns the activation for the
       radial basis functions specified by (`p`, `v`, `sigma`) calculated at
       `s`. `p` is a list of position centers, `v` is a list of velocity centers,
       and `sigma` is the basis function width. The return value f is a vector
       with activations.
       
       `s` is a vector containing the state, or may be a matrix in which each
       row specifies a state. In that case, `f` is a matrix where each row
       contains the activation for a row in `s`.
    """
       
    s = np.atleast_2d(s)
    pd = np.arctan2(s[:, None, 1], s[:, None, 0]) - p.flatten()
    pd = abs((pd-pi)%(2*pi)-pi)
    
    dist = np.sqrt(pd**2 + ((s[:, None, 2] - v.flatten())/(8/pi))**2)
    return np.squeeze(scipy.stats.norm.pdf(dist, 0, sigma))

def rbfprojector(nbasis, sigma):
    """Returns function that projects states onto Gaussian radial basis function features.
    
       feature = rbfprojector(nbasis, sigma) returns a function
           f = feature(s)
       that projects a state `s` onto a Gaussian RBF feature vector `f`. `nbasis` is the number
       of basis functions per dimension, while `sigma` is their width.
       
       If `s` is a matrix where each row is a state, `f` is a matrix where each row
       contains the feature vector for a row of `s`.
       
       EXAMPLE
           >>> feature = rbfprojector(3, 2)
           >>> print(feature([0, 0, 0]))
           [0.01691614 0.05808858 0.05808858 0.19947114 0.01691614 0.05808858]
    """

    p, v = np.meshgrid(np.linspace(-pi, pi-(2*pi)/(nbasis-1), nbasis-1), np.linspace(-8, 8, nbasis))
    return lambda x: __gaussrbf(x, p, v, sigma)

class DDPG:
    """Deep Deterministic Policy Gradient

       METHODS
           train       -- Train network.
           __call__    -- Evaluate network.
           __ilshift__ -- Copy network weights.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25], tau=0.01):
        """Creates a new DDPG network.
        
           DDPG(states, actions) creates a DDPG network with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation. The last actor layer uses the
           hyperbolic tangent. As such, all actions are scaled to [-1, 1]."
           
           DDPG(states, actions, hiddens) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> ddpg = DDPG(2, 1, [10, 10])
        """
        
        self.__tau = tau

        # Actor
        inputs = tf.keras.Input(shape=(states,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(1, activation='tanh')(layer)
        self.__actor = tf.keras.Model(inputs, outputs)
        self.__opt = tf.keras.optimizers.Adam()
        
        # Critic
        inputs = tf.keras.Input(shape=(states+actions,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

        self.__critic = tf.keras.Model(inputs, outputs)
        self.__critic.compile(loss=tf.keras.losses.MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam())

    def train(self, s, a, target):
        """Trains both critic and actor.
        
           DDPG.train(s, a, target) trains the critic such that
           it approaches DDPG.critic(s, a) = target, and the actor to
           approach DDPG.actor(s) = max_a'(DDPG.critic(s, a'))
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> ddpg.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], [12, 16, 19])
        """
        
        # Critic
        inp, reshape = self.__combine(s, a)
        self.__critic.train_on_batch(inp, np.atleast_1d(target))
        
        # Actor
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q = -self.__critic(tf.concat([s, self.__actor(s)], 1))
        grad = tape.gradient(q, self.__actor.variables)
        self.__opt.apply_gradients(zip(grad, self.__actor.variables))
        
    def critic(self, s, a=None):
        """Evaluates the value function (critic).
        
           DDPG.critic(s) returns the value of the approximator at observation
           `s` and the actor's action.

           DDPG.critic(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.
           
           EXAMPLE
               >>> ddpg = DQN(2, 1)
               >>> # single observation and action
               >>> print(ddpg.critic([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(ddpg.critic([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(ddpg.critic([0.1, 2], [-1, 1]))
               [[12  -12]]
        """
        
        if a is None:
            s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
            out = self.__critic.predict_on_batch(tf.concat([s, self.__actor(s)], 1))
            return out
        else:
            inp, reshape = self.__combine(s, a)
            out = np.asarray(self.__critic.predict_on_batch(inp))
            if reshape:
                out = np.reshape(out, reshape)
            return out

    def actor(self, s):
        """Evaluates the policy(actor).
        
           DDPG.actor(s) returns the action to take in state `s`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           
           EXAMPLE
               >>> ddpg = DDPG(2, 1)
               >>> # single observation
               >>> print(ddpg.actor([0.1, 2]))
               [-0.23]
               >>> # batch of observations
               >>> print(dqn([[0.1, 2], [0.4, 3]]))
               [[-0.23]
                [0.81]]
        """
        
        squeeze = False
        if len(s.shape) == 1:
            squeeze = True

        s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
        out = self.__actor.predict_on_batch(s)

        if squeeze:
            out = np.atleast_1d(np.squeeze(out))

        return out
        
    def __ilshift__(self, other):
        """Copies network weights.
        
           dqn2 <<= dqn1 copies the weights from `dqn1` into `dqn2`. The
           networks must have the same structure.
        """

        self.__critic.set_weights(other.__critic.get_weights())
        self.__actor.set_weights(other.__actor.get_weights())

        return self

    def __combine(self, s, a):
        # Massage s into a 2d array of type float32
        s = np.atleast_2d(np.asarray(s, dtype=np.float32))

        # Massage a into 2d "row-array" of type float32
        a = np.atleast_2d(np.asarray(a, dtype=np.float32))
        if a.shape[1] > 1:
            a = a.transpose()

        # Replicate s and a if necessary
        reshape = None
        if s.shape[0] == 1 and a.shape[0] > 1:
            reshape = [1, a.shape[0]]
            s = np.tile(s, [a.shape[0], 1])
        elif s.shape[0] > 1 and a.shape[0] > 1 and s.shape[0] != a.shape[0]:
            reshape = [s.shape[0], a.shape[0]]
            s = np.repeat(s, np.repeat(reshape[1], reshape[0]), axis=0)
            a = np.tile(a, [reshape[0], 1])

        inp = np.hstack((s, a)).astype(np.float32)

        return inp, reshape

class Memory:
    """Replay memory
       
       METHODS
           add    -- Add transition to memory.
           sample -- Sample minibatch from memory.
    """
    def __init__(self, states, actions, size=1000000):
        """Creates a new replay memory.
        
           Memory(states, action) creates a new replay memory for storing
           transitions with `states` observation dimensions and `actions`
           action dimensions. It can store 1000000 transitions.
           
           Memory(states, actions, size) additionally specifies how many
           transitions can be stored.
        """

        self.s = np.ndarray([size, states])
        self.a = np.ndarray([size, actions])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, states])
        self.done = np.ndarray([size, 1])
        self.n = 0
    
    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""

        return self.n
    
    def add(self, s, a, r, sp, done):
        """Adds a transition to the replay memory.
        
           Memory.add(s, a, r, sp, done) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `done`
           specifies whether the episode finished at state `sp`.
        """

        self.s[self.n, :] = s
        self.a[self.n, :] = a
        self.r[self.n, :] = r
        self.sp[self.n, :] = sp
        self.done[self.n, :] = done
        self.n += 1
    
    def sample(self, size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.done[idx]

"""OpenAI Gym Environment wrapper.

   METHODS
       reset   -- Reset environment
       step    -- Step environment
       render  -- Visualize environment
       close   -- Close visualization
       
   MEMBERS
       states  -- Number of state dimensions
       actions -- Number of action dimensions
"""
class Environment():
    def reset(self):
        """Reset environment to start state.
        
           obs = env.reset() returns the start state observation.
        """
        return self.env.reset()
    
    def step(self, u):
        """Step environment.
        
           obs, r, done, info = env.step(u) takes action u and
           returns the next state observation, reward, whether
           the episode terminated, and extra information.
        """
        return self.env.step(u)
    
    def render(self):
        """Render environment.
        
           env.render() renders the current state of the
           environment in a separate window.
           
           NOTE
               You must call env.close() to close the window,
               before creating a new environment; otherwise
               the kernel may hang.
        """
        return self.env.render()
    
    def close(self):
        """Closes the rendering window."""
        return self.env.close()    

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
            