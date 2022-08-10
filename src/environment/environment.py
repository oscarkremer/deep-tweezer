
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