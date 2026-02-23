import numpy as np


class LIF:
    """
    Leaky Integrate-and-Fire neuron.

    Args:
            decay: Membrane decay (subtraction based)
            threshold: Magnutude of membrane potential needed to produce a spike
            reset: Value the membrane potential is reset to after spike
    """

    def __init__(self, decay=0.2, threshold=2.0, reset=0.0):
        # --- Neuron dynamics ---
        self.decay = decay              # Membrane decay
        self.threshold = threshold
        self.reset = reset
        self.mem = 0.0
        self.spk = 0

    def update(self, synaptic_input):
        """Membrane update. Returns spike (0 or 1)."""
        self.spk = 0
        self.mem = max(0.0, self.mem - self.decay + synaptic_input)

        if self.mem >= self.threshold:
            self.spk = 1
            self.mem = self.reset

        return self.spk


class RSTDPSynapse:
    """
    Reward-modulated STDP synapse with rectangular window.

    Args:
            learning_rate: Scales the weight update
            w_init: Initial weight (random if None)
            t_pre/t_post: Rectangular STDP window widths (timesteps)
                          Pre-before-post within t_pre → LTP (causal)
                          Post-before-pre within t_post → LTD (acausal)
            tau_e_shift: Eligibility decay as right-shift (divide by 2^N each step)
                         Higher = slower decay, longer credit assignment window
            dw_pos/dw_neg: Fixed weight increment/decrement on spike pairing
                           Equivalent to A_plus/A_minus in exponential STDP
            w_min/w_max: Weight clamps
    """

    # Eligibility traces start as disabled/inactive
    DISABLED = -1

    def __init__(self, learning_rate=0.1, w_init=None,
                 t_pre=5, t_post=5, tau_e_shift=4,
                 dw_pos=0.125, dw_neg=0.125,
                 w_min=0.05, w_max=1.0):

        self.learning_rate = learning_rate
        self.weight = w_init if w_init is not None else np.random.uniform(0.3, 0.8)

        # STDP window parameters
        self.t_pre = t_pre
        self.t_post = t_post
        self.tau_e_shift = tau_e_shift
        self.dw_pos = dw_pos
        self.dw_neg = dw_neg

        # Counter-based trace state (pre/post_timer = -1 -> inactive, 0+ = counting)
        self.pre_timer = self.DISABLED
        self.post_timer = self.DISABLED
        self.eligibility = 0.0

        # Weight bounds
        self.w_min = w_min
        self.w_max = w_max

    def update_traces_and_eligibility(self, pre_spike, post_spike):
        """
        Update spike timing counters and eligibility each timestep.
        
        Uses rectangular STDP windows: if a spike pair occurs within
        the window, eligibility is incremented/decremented by dw_pos/dw_neg.
        
        Args:
            pre_spike: 1 if pre-synaptic neuron fired, 0 otherwise
            post_spike: 1 if post-synaptic neuron fired, 0 otherwise  
        """
        # STDP causality timers
        if self.pre_timer >= 0:
            self.pre_timer += 1
            if self.pre_timer > self.t_pre:
                self.pre_timer = self.DISABLED

        if self.post_timer >= 0:
            self.post_timer += 1
            if self.post_timer > self.t_post:
                self.post_timer = self.DISABLED

        # STDP on pre-spike: start pre timer, check for acausality (LTD)
        if pre_spike:
            if self.post_timer >= 0 and self.post_timer <= self.t_post:
                self.eligibility -= self.dw_neg
                self.post_timer = self.DISABLED
            self.pre_timer = 0

        # STDP on post-spike: start post timer, check for causality (LTP)
        if post_spike:
            if self.pre_timer >= 0 and self.pre_timer <= self.t_pre:
                self.eligibility += self.dw_pos
                self.pre_timer = self.DISABLED
            self.post_timer = 0

        # Decay eligibility via right-shift (FPGA: subtract eligibility >> tau_e_shift)
        self.eligibility = self.eligibility - self.eligibility / (1 << self.tau_e_shift)

    def apply_reward(self, dopamine):
        """
        Apply reward-modulated weight update.
        
        Args:
            dopamine: Reward signal. Positive reinforces correlated activity, negative punishes it.
        """
        delta_w = self.learning_rate * dopamine * self.eligibility
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)


class SNNLayer:
    """
    A fully-connected SNN with STDP/R-STDP learning.

    Args:
            n_input: Number of pre-synaptic input neurons
            n_output: Number of post-synaptic output neurons
            neuron_params: Dict passed to LIF constructor
            synapse_params: Dict passed to RSTDPSynapse constructor
    """

    def __init__(self, n_inputs, n_outputs, neuron_params=None, synapse_params=None):
        neuron_params = neuron_params or {}
        synapse_params = synapse_params or {}

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Create output neurons
        self.neurons = [LIF(**neuron_params) for _ in range(n_outputs)]

        # Create synapse matrix: synapses[post]x[pre]
        self.synapses = [
            [RSTDPSynapse(**synapse_params) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]

    def forward(self, input_spikes):
        """
        Process one timestep/frams.
        
        Args:
            input_spikes: List/array of length n_input (0s and 1s)
            
        Returns:
            List of output spikes (length n_output)
        """
        output_spikes = []

        for j, neuron in enumerate(self.neurons):
            # Compute weighted synaptic input
            synaptic_input = sum(
                self.synapses[j][i].weight * input_spikes[i]
                for i in range(self.n_inputs)
            )

            # Update neuron
            spike = neuron.update(synaptic_input)
            output_spikes.append(spike)

            # Update all incoming synapse traces and eligibility
            for i in range(self.n_inputs):
                self.synapses[j][i].update_traces_and_eligibility(
                    pre_spike=input_spikes[i],
                    post_spike=spike,
                )

        return output_spikes

    def winner_takes_all(self, output_spikes):
        """
        Returns index of winning neuron.
        Spiking neurons always beat non-spiking ones.
        Magnitude of membrane potential breaks ties.
        """
        spiking = [i for i, s in enumerate(output_spikes) if s == 1]

        if len(spiking) == 1:
            return spiking[0]
        elif len(spiking) > 1:
            return spiking[0]
        else:
            # No spikes: highest membrane potential
            return int(np.argmax([n.mem for n in self.neurons]))

    def apply_reward(self, dopamine, winner_idx):
        """Apply reward only to the winning neuron's synapses."""
        for i in range(self.n_inputs):
            # Reinforce/punish winner
            self.synapses[winner_idx][i].apply_reward(dopamine)

    def get_weights(self):
        """Return weight matrix as numpy array [n_output x n_input]."""
        return np.array([
            [self.synapses[j][i].weight for i in range(self.n_inputs)]
            for j in range(self.n_outputs)
        ])

    def reset_state(self):
        """Reset the state of all neurons in the network."""
        for n in self.neurons:
            n.mem = 0.0
            n.spk = 0