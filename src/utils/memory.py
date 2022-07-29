
import numpy as np


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

