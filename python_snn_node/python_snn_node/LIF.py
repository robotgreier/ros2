import numpy as np

class LIF():
    def __init__(self, 
                beta=0.9, 
                threshold=2.0, 
                reset=0.0, 
                learning_rate=0.1, 
                eligibility_decay=0.9, 
                wmin=0.01, 
                wmax=1.0):
        self.beta = beta
        self.threshold = threshold
        self.mem = 0.0
        self.reset = reset
        self.learning_rate = learning_rate
        self.spk = 0
        self.eligibility = 0.0
        self.eligibility_decay = eligibility_decay
        # nytt: grensene brukes i STDP/rSTDP
        self.wmin = float(wmin)
        self.wmax = float(wmax)


    def update(self, synaptic_input):
        # Reset spike
        self.spk = 0

        # Update eligibility trace
        self.eligibility = (self.eligibility_decay * self.eligibility + synaptic_input)
        self.eligibility = np.clip(self.eligibility, -10, 10)

        # Update membrane potential
        self.mem = self.beta * self.mem + synaptic_input

        # Check for spike
        if self.mem > self.threshold:
            self.spk = 1
            self.mem = self.reset
            self.eligibility = self.reset
        
        return self.spk

    def STDP(self, weight, pre_eligibility, is_winner):
        # Standard STDP for offline/unsupervised learning
        timing_factor = np.clip(pre_eligibility * self.eligibility, -10, 10)
        
        if is_winner:
            weight += self.learning_rate * timing_factor
        else:
            weight -= self.learning_rate * timing_factor * 0.5
                
        return np.clip(weight, self.wmin, self.wmax)

    def rSTDP(self, weight, pre_eligibility, is_winner, dopamine):
        # Reward-modulated STDP for online learning with reward signal
        timing_factor = np.clip(pre_eligibility * self.eligibility, -10, 10)
        effective_lr = self.learning_rate / (1 + abs(timing_factor) / 200)
        
        if is_winner:
            weight += effective_lr * timing_factor * dopamine
        else:
            if dopamine > 0:
                weight -= effective_lr * timing_factor * 4.0
            else:
                weight -= effective_lr * timing_factor * 0.1
                
        return np.clip(weight, self.wmin, self.wmax)