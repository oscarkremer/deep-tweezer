"""ELE2761 Deep Reinforcement Learning helper functinos

Implements the provided functionality to be used in your solution.

CLASSES
    Model        -- Dynamics model approximator
    Memory       -- Replay Memory
    Pendulum     -- OpenAI Gym Pendulum-v0 environment

FUNCTIONS
    rbfprojector -- Gaussian RBF projector factory
"""

from math import pi
import numpy as np
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt
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


"""Base functions for Deep RL networks

   METHODS
       __ilshift__ -- Copy network weights.
       combine     -- Combine state and action vectors.
"""
class Network:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
    
    def __ilshift__(self, other):
        """Copies network weights.
        
           network2 <<= network1 copies the weights from `network1` into `network2`. The
           networks must have the same structure.
        """

        if isinstance(self, DQN) or isinstance(self, Model):
            self.__model.set_weights(other.__model.get_weights())
            
        if isinstance(self, DDPG):
            self._DDPG__actor.set_weights(other._DDPG__actor.get_weights())
            self._DDPG__critic.set_weights(other._DDPG__critic.get_weights())

        return self

    def combine(self, s, a, force=False):
        """Combines state and action vectors into single network input.
        
           m, reshape = Network.combine(s, a) has five cases. In all cases,
           `m` is a matrix and `reshape` is a shape to which the network Q output
           should be reshaped. The shape will be such that states are in 
           rows and actions are in columns of `m`.
           
            1) `s` and `a` are vectors. They will be concatenated.
            2) `s` is a matrix and `a` is a vector. `a` will be replicated for
               each `s`.
            3) `s` is a vector and `a` is a matrix. `s` will be replicated for
               each `a`.
            4) `s` and `a` are matrices with the same number of rows. They will
               be concatenated.
            5) `s` and `a` are matrices with different numbers of rows or
               force=True. Each `s` will be replicated for each `a`.
              
           EXAMPLE
               >>> print(network.combine([1, 2], 5))
               (array([[1., 2., 5.]], dtype=float32), (1, 1))
               >>> print(network.combine([[1, 2], [3, 4]], 5))
               (array([[1., 2., 5.],
                       [3., 4., 5.]], dtype=float32), (2, 1))
               >>> print(network.combine([1, 2], [5, 6])) # single action only
               (array([[1., 2., 5.],
                       [1., 2., 6.]], dtype=float32), (1, 2))
               >>> print(network.combine([1, 2], [[5], [6]]))
               (array([[1., 2., 5.],
                      [1., 2., 6.]], dtype=float32), (1, 2))
               >>> print(network.combine([[1, 2], [3, 4]], [5, 6])) # single action only
               (array([[1., 2., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 1))
               >>> print(network.combine([[1, 2], [3, 4]], [[5], [6]]))
               (array([[1., 2., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 1))
               >>> print(network.combine([[1, 2], [3, 4]], [[5], [6]], force=True))
               (array([[1., 2., 5.],
                       [1., 2., 6.],
                       [3., 4., 5.],
                       [3., 4., 6.]], dtype=float32), (2, 2))
        """
        
        # Convert scalars to vectors
        s = np.atleast_1d(np.asarray(s, dtype=np.float32))
        a = np.atleast_1d(np.asarray(a, dtype=np.float32))
        
        # Convert vectors to matrices for single-state environments
        if self.states == 1 and len(s.shape) == 1 and s.shape[0] > 1:
            s = np.atleast_2d(s).transpose()
            
        # Convert vectors to matrices for single-action environments
        if self.actions == 1 and len(a.shape) == 1 and a.shape[0] > 1:
            a = np.atleast_2d(a).transpose()

        # Normalize to matrices
        s = np.atleast_2d(s)
        a = np.atleast_2d(a)

        # Sanity checking
        if len(s.shape) > 2 or len(a.shape) > 2:
            raise ValueError("Input dimensionality not supported")
        
        if s.shape[1] != self.states:
            raise ValueError("State dimensionality does not match network")
            
        if a.shape[1] != self.actions:
            raise ValueError("Action dimensionality does not match network")
            
        # Replicate if necessary
        if s.shape[0] != a.shape[0] or force:
            reshape = (s.shape[0], a.shape[0])
            s = np.repeat(s, np.repeat(reshape[1], reshape[0]), axis=0)
            a = np.tile(a, (reshape[0], 1))
        else:
            reshape = (s.shape[0], 1)

        m = np.hstack((s, a))

        return m, reshape

class DQN(Network):
    """Deep learning-based Q approximator.

       METHODS
           train       -- Train network.
           __call__    -- Evaluate network.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25]):
        """Creates a new Q approximator.
        
           DQN(states, actions) creates a Q approximator with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation."
           
           DQN(states, actions, hiddens) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> dqn = DQN(2, 1, [10, 10])
        """
        
        super(DQN, self).__init__(states, actions)
        
        inputs = tf.keras.Input(shape=(states+actions,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(1, activation='linear')(layer)

        self.__model = tf.keras.Model(inputs, outputs)
        self.__model.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=tf.keras.optimizers.Adam())

    def train(self, s, a, target):
        """Trains the Q approximator.
        
           DQN.train(s, a, target) trains the Q approximator such that
           it approaches DQN(s, a) = target.
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a vector specifying an
           action for every observation in the batch. `target` is a vector
           specifying a target value for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> dqn.train([[0.1, 2], [0.4, 3], [0.2, 5]], [-1, 1, 0], [12, 16, 19])
        """
           
        inp, reshape = self.combine(s, a)
        self.__model.train_on_batch(inp, np.atleast_1d(target))

    def __call__(self, s, a):
        """Evaluates the Q approximator.
        
           DQN(s, a) returns the value of the approximator at observation
           `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           If `a` is the same size as the number of rows in `s`, it specifies
           the action at which to evaluate each observation in the batch.
           Otherwise, it specifies the action(s) at which the evaluate ALL
           observations in the batch.
           
           EXAMPLE
               >>> dqn = DQN(2, 1)
               >>> # single observation and action
               >>> print(dqn([0.1, 2], -1))
               [[ 12 ]]
               >>> # batch of observations and actions
               >>> print(dqn([[0.1, 2], [0.4, 3]], [-1, 1]))
               [[12]
                [16]]
               >>> # evaluate single observation at multiple actions
               >>> print(dqn([0.1, 2], [-1, 1]))
               [[12  -12]]
        """

        inp, reshape = self.combine(s, a)
        return np.reshape(np.asarray(self.__model(inp)), reshape)

    def __ilshift__(self, other):
        """Copies network weights.
        
           network2 <<= network1 copies the weights from `network1` into `network2`. The
           networks must have the same structure.
        """

        self.__model.set_weights(other.__model.get_weights())

        return self
    
class DDPG(Network):
    """Deep Deterministic Policy Gradient

       METHODS
           train       -- Train network.
           critic      -- Evaluate critic network.
           actor       -- Evaluate actor network.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25]):
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
        
        super(DDPG, self).__init__(states, actions)

        # Actor
        inputs = tf.keras.Input(shape=(states,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(actions, activation='tanh')(layer)
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
        inp, reshape = self.combine(s, a)
        self.__critic.train_on_batch(inp, np.atleast_1d(target))
        
        # Actor
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q = -self.__critic.call(tf.concat([s, self.__actor(s)], 1))
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
            out = self.__critic(tf.concat([s, self.__actor(s)], 1)).numpy()
            return out
        else:
            inp, reshape = self.combine(s, a)
            return np.reshape(np.asarray(self.__critic(inp)), reshape)

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
        
        single = len(np.asarray(s).shape) == 1

        s = tf.convert_to_tensor(np.atleast_2d(s), dtype=tf.float32)
        out = self.__actor(s).numpy()

        if single:
            out = out[0]

        return out

    def __ilshift__(self, other):
        self._DDPG__actor.set_weights(other._DDPG__actor.get_weights())
        self._DDPG__critic.set_weights(other._DDPG__critic.get_weights())

        return self
    
    
class Model(Network):
    """Deep learning-based dynamics model approximator.

       METHODS
           train       -- Train network on minibatch.
           fit         -- Fit network on memory.
           __call__    -- Evaluate network.
    """

    def __init__(self, states, actions=1, hiddens=[25, 25]):
        """Creates a new dynamics model approximator.
        
           Model(states, actions) creates a dynamics approximator with `states`
           observation dimensions and `actions` action dimensions. It has
           two hidden layers with 25 neurons each. All layers except
           the last use ReLU activation."
           
           Model(states, actions, hiddens) additionally specifies the
           number of neurons in the hidden layers.

           EXAMPLE
               >>> model = Model(2, 1, [10, 10])
        """
        
        super(Model, self).__init__(states, actions)
        
        inputs = tf.keras.Input(shape=(states+actions,))
        layer = inputs
        for h in hiddens:
            layer = tf.keras.layers.Dense(h, activation='relu')(layer)
        outputs = tf.keras.layers.Dense(states+1, activation='linear')(layer)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=tf.keras.optimizers.Adam())
        
        self.diff = False

    def train(self, s, a=None, r=None, sp=None):
        """Trains the dynamics approximator.
        
           Model.train(s, a, sp) trains the dynamics approximator such that
           it approaches Model(s, a) = r, sp.
           
           `s` is a matrix specifying a batch of observations, in which
           each row is an observation. `a` is a matrix specifying an
           action for every observation in the batch. `sp` is a matrix
           specifying the next state for each observation-action pair in
           the batch.
           
           EXAMPLE
               >>> bs, ba, br, bsp, bd = memory.sample(batch)
               >>> model.fit(bs, ba, bsp)
        """
        
        r = np.atleast_2d(r)
        if r.shape[1] > 1:
            r = r.T

        inp, reshape = self.combine(s, a)
        if self.diff:
            self.model.train_on_batch(inp, np.hstack((np.atleast_2d(sp-s), r)))
        else:
            self.model.train_on_batch(inp, np.hstack((np.atleast_2d(sp), r)))

    def fit(self, memory):
        """Fits the dynamics approximator.
        
           Model.fit(memory) fits the dynamics approximator such that it
           approaches Model(s, a) = r, sp for all `s`, `a`, `r`, `sp` in `memory`.

           EXAMPLE
               >>> model.fit(memory)
               Epoch 00393: early stopping
               Model validation loss  0.003967171715986397
        """
        
        p = np.random.permutation(len(memory))
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
        
        if self.diff:
            history = self.model.fit(np.hstack((memory.s[p], memory.a[p])), np.hstack((memory.sp[p]-memory.s[p], memory.r[p])), verbose=0, validation_split=0.1, epochs=1000, callbacks=[es])
        else:
            history = self.model.fit(np.hstack((memory.s[p], memory.a[p])), np.hstack((memory.sp[p], memory.r[p])), verbose=0, validation_split=0.1, epochs=1000, callbacks=[es])
        
        #print('Model validation loss ', history.history['loss'][-1])
        
    def __call__(self, s, a):
        """Evaluates the dynamics approximator.
        
           r, sp = Model(s, a) returns the approximated reward `r` and next
           state `sp` at observation `s` and action `a`.
           
           `s` is either a vector specifying a single observation, or a
           matrix in which each row specifies one observation in a batch.
           `a` is the same size as the number of rows in `s`, and specifies
           the action at which to evaluate each observation in the batch.
           
           EXAMPLE
               >>> model = Model(2, 1, [10, 10])
               >>> print(model([0, 0], 0))
               (0.0, [0., 0.])
               >>> print(model([[0, 0], [1, 1]], [0, 1]))
               ([0.        , 0.12079704], [[ 0.        ,  0.        ],
                                           [-0.04185265,  0.16854128]])
        """

        inp, reshape = self.combine(s, a)
        if reshape[0] != inp.shape[0] or reshape[1] != 1:
            raise ValueError("Input does not describe one action per state")
            
        reshape = (reshape[0], self.states+1)
        out = np.reshape(np.asarray(self.model(inp)), reshape)
        
        r = out[:,-1]
        if self.diff:
            sp = out[:, :-1] + np.atleast_2d(s)
        else:
            sp = out[:, :-1]
            
        if len(np.asarray(s).shape) == 1:
            r = r[0]
            sp = sp[0]
        
        return r, sp

    def __ilshift__(self, other):
        self.__model.set_weights(other.__model.get_weights())

        return self
    
    
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
        self.actions = self.env.action_space.shape[0]

    def step(self, u):
        """Step environment."""
        return self.env.step(np.atleast_1d(u))
    
    def normalize(self, s):
        """Normalize state to unit circle.
        
           s = env.normalize(s) normalizes `s` such that its cosine-sine
           angle representation falls on the unit circle.
           
           EXAMPLE
               >>> env = Pendulum()
               >>> print(env.normalize([1, 1, 2])
               [0.70710678 0.70710678 2.        ]
        """
        
        single = len(np.asarray(s).shape) == 1
        
        s = np.atleast_2d(s)
        ang = np.arctan2(s[:,None,1], s[:,None,0])
        s = np.hstack((np.cos(ang), np.sin(ang), s[:,None,2]))
        
        if single:
            s = s[0]
            
        return s

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
    
    def plotnetwork(self, network):
        """Plot network.

           plot(dqn) plots the value function and induced policy of DQN network `dqn`.
           plot(ddpg) plots the value function and policy of DDPG network `ddpg`.
           plot(model) plots the dynamics approximation of Model network `model`.
        """
        if network.states != 3 or network.actions != 1:
            raise ValueError("Network is not compatible with Pendulum environment")

        pp, vv = np.meshgrid(np.linspace(-np.pi,np.pi, 64), np.linspace(-8, 8, 64))
        obs = np.hstack((np.reshape(np.cos(pp), (pp.size, 1)),
                         np.reshape(np.sin(pp), (pp.size, 1)),
                         np.reshape(       vv , (vv.size, 1))))

        aval = np.linspace(-2, 2, 3)

        if isinstance(network, DQN):
            qq = network(obs, aval)
            vf = np.reshape(np.amax(qq, axis=1), pp.shape)
            pl = np.vectorize(lambda x: aval[x])(np.reshape(np.argmax(qq, axis=1), pp.shape))

            fig, axs = plt.subplots(1,2)
            fig.subplots_adjust(right=1.5)

            h = axs[0].contourf(pp, vv, vf, 256)
            fig.colorbar(h, ax=axs[0])
            h = axs[1].contourf(pp, vv, pl, 256)
            fig.colorbar(h, ax=axs[1])

            axs[0].set_title('Value function')
            axs[1].set_title('Policy')
        elif isinstance(network, DDPG):
            vf = np.reshape(network.critic(obs), pp.shape)
            pl = np.reshape(network.actor(obs), pp.shape)

            fig, axs = plt.subplots(1,2)
            fig.subplots_adjust(right=1.5)

            h = axs[0].contourf(pp, vv, vf, 256)
            fig.colorbar(h, ax=axs[0])
            h = axs[1].contourf(pp, vv, pl, 256)
            fig.colorbar(h, ax=axs[1])

            axs[0].set_title('Critic')
            axs[1].set_title('Actor')
        elif isinstance(network, Model):
            fig, axs = plt.subplots(len(aval), 3)
            fig.subplots_adjust(top=2,right=1.5)

            for aa in range(len(aval)):
                r, sp = network(obs, aval[aa])
                pd = np.reshape(np.arctan2(sp[:, 1], sp[:,0]), pp.shape)-pp
                vd = np.reshape(sp[:,2], vv.shape)-vv
                r = np.reshape(r, pp.shape)

                h = axs[aa, 0].contourf(pp, vv, pd, np.linspace(-0.5, 0.5, 256))
                fig.colorbar(h, ax=axs[aa, 0])
                h = axs[aa, 1].contourf(pp, vv, vd, np.linspace(-1.75, 1.75, 256))
                fig.colorbar(h, ax=axs[aa, 1])
                h = axs[aa, 2].contourf(pp, vv, r, 256)
                fig.colorbar(h, ax=axs[aa, 2])

                axs[aa, 0].set_title('Position derivative (a={})'.format(aval[aa]))
                axs[aa, 1].set_title('Velocity derivative (a={})'.format(aval[aa]))
                axs[aa, 2].set_title('Reward (a={})'.format(aval[aa]))
        else:
            raise ValueError("Input should be either DQN, DDPG or Model, not {}".format(type(network).__name__))
