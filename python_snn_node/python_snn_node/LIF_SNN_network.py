import numpy as np


class LIF:
    """
    Leaky Integrate-and-Fire neuron.

    Args:
            decay: Membrane decay (subtraction based)
            threshold: Magnitude of membrane potential needed to produce a spike
            reset: Value the membrane potential is reset to after spike
    """

    def __init__(self, decay=0.75, threshold=4.0, reset=0.0):
        # --- Neuron dynamics ---
        self.decay = decay              # Membrane decay
        self.threshold = threshold
        self.reset = reset
        self.mem = 0.0
        self.pre_reset_mem = 0.0       # Membrane potential before reset, used for WTA
        self.spk = 0

    def update(self, synaptic_input):
        """Membrane update. Returns spike (0 or 1)."""
        self.spk = 0
        self.mem = self.mem - self.decay + synaptic_input
        self.pre_reset_mem = self.mem  # Cache membrane potential before reset for WTA

        if self.mem >= self.threshold:
            self.spk = 1
            self.mem = self.reset

        return self.spk


class RSTDPSynapse:
    """
    Reward-modulated STDP synapse with rectangular window.

    Supports two modes:
        'rstdp': Weight updates only when apply_reward() is called externally
                 with a dopamine signal. Eligibility trace accumulates STDP
                 events and decays over time, acting as a credit assignment window.
        'stdp':  Weight updates immediately on each spike pairing (dopamine=1.0).
                 Eligibility still decays each step, so the effective update is
                 lr * eligibility at the moment of each spike — not a pure
                 instantaneous step, but a short-window trace-based update.

    Args:
            mode: 'rstdp' (default) or 'stdp'
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

    def __init__(self, learning_rate=0.125, w_init=0.3,
                 t_pre=2, t_post=3, tau_e_shift=4,
                 dw_pos=0.25, dw_neg=0.03125,
                 w_min=0.03125, w_max=1.0,
                 mode='rstdp'):

        self.mode = mode
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

    def update_eligibility(self, pre_spike, post_spike):
        """
        Update spike timing counters and eligibility each timestep.

        Uses rectangular STDP windows: if a spike pair occurs within
        the window, eligibility is incremented/decremented by dw_pos/dw_neg.
        Timers run independently so multiple pairings within a window
        are detected — equivalent to parallel trace registers in HDL.

        In 'stdp' mode, weight is updated immediately after eligibility update
        (dopamine=1.0), so no external apply_reward() call is needed.

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
            self.pre_timer = 0

        # STDP on post-spike: start post timer, check for causality (LTP)
        if post_spike:
            if self.pre_timer >= 0 and self.pre_timer <= self.t_pre:
                self.eligibility += self.dw_pos
            self.post_timer = 0

        # Decay eligibility via right-shift
        self.eligibility = self.eligibility - self.eligibility / (1 << self.tau_e_shift)

        # Clamp eligibility
        self.eligibility = np.clip(self.eligibility, -1.0, 1.0)

        # In stdp mode, apply weight update immediately
        if self.mode == 'stdp':
            self.apply_reward(dopamine=1.0)

    def apply_reward(self, dopamine):
        """
        Apply reward-modulated weight update.

        Args:
            dopamine: Reward signal. Positive reinforces correlated activity, negative punishes it.
                      In 'stdp' mode this is always 1.0 (called internally).
        """
        delta_w = self.learning_rate * dopamine * self.eligibility
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)


class SNNLayer:
    """
    A fully-connected SNN with STDP/R-STDP learning.

    Args:
            n_inputs: Number of pre-synaptic input neurons
            n_outputs: Number of post-synaptic output neurons
            neuron_params: Dict passed to LIF constructor
            synapse_params: Dict passed to RSTDPSynapse constructor
                            Include 'mode': 'stdp' or 'rstdp' (default)
    """

    def __init__(self, n_inputs, n_outputs, neuron_params=None, synapse_params=None):
        neuron_params = neuron_params or {}
        synapse_params = synapse_params or {}

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.mode = synapse_params.get('mode', 'rstdp')

        # Create output neurons
        self.neurons = [LIF(**neuron_params) for _ in range(n_outputs)]

        # Create synapse matrix: synapses[post][pre]
        self.synapses = [
            [RSTDPSynapse(**synapse_params) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]

    def forward(self, input_spikes):
        """
        Process one timestep/frame.

        Args:
            input_spikes: List/array of length n_inputs (0s and 1s)

        Returns:
            List of output spikes (length n_outputs)
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

        # Update synapses after all neurons have been evaluated
        if self.mode == 'stdp':
            # Lateral inhibition: only winner's synapses receive LTP/LTD,
            # but all eligibility traces decay each step
            winner = self.winner_takes_all(output_spikes)
            for j in range(self.n_outputs):
                if j != winner:
                    self.neurons[j].mem = self.neurons[j].reset  # suppress losers
            for j in range(self.n_outputs):
                for i in range(self.n_inputs):
                    # Winner gets full pre/post spike events; losers only decay
                    pre = input_spikes[i] if j == winner else 0
                    post = output_spikes[j] if j == winner else 0
                    self.synapses[j][i].update_eligibility(
                        pre_spike=pre,
                        post_spike=post,
                    )
        else:
            # R-STDP: update all synapses, weight updates happen externally via apply_reward()
            for j in range(self.n_outputs):
                for i in range(self.n_inputs):
                    self.synapses[j][i].update_eligibility(
                        pre_spike=input_spikes[i],
                        post_spike=output_spikes[j],
                    )

        return output_spikes

    def winner_takes_all(self, output_spikes):
        """
        Returns index of winning neuron.
        Spiking neurons always beat non-spiking ones.
        Pre-reset membrane potential breaks ties, ensuring index-independent
        results equivalent to a parallel hardware comparator on FPGA.
        """
        spiking = [i for i, s in enumerate(output_spikes) if s == 1]

        if len(spiking) == 1:
            return spiking[0]
        elif len(spiking) > 1:
            # Multiple spikes: highest pre-reset membrane potential wins
            return max(spiking, key=lambda i: self.neurons[i].pre_reset_mem)
        else:
            # No spikes: highest pre-reset membrane potential
            return int(np.argmax([n.pre_reset_mem for n in self.neurons]))

    def apply_reward(self, dopamine, winner_idx):
        """
        Apply reward only to the winning neuron's synapses.
        No-op in 'stdp' mode (weights already updated in forward()).
        """
        if self.mode == 'stdp':
            return
        for i in range(self.n_inputs):
            # Reinforce/punish winner
            self.synapses[winner_idx][i].apply_reward(dopamine)

    def get_weights(self):
        """Return weight matrix as numpy array [n_outputs x n_inputs]."""
        return np.array([
            [self.synapses[j][i].weight for i in range(self.n_inputs)]
            for j in range(self.n_outputs)
        ])

    def reset_state(self):
        """Reset the state of all neurons and synaptic traces in the network."""
        for n in self.neurons:
            n.mem = 0.0
            n.spk = 0
            n.pre_reset_mem = 0.0
        for j in range(self.n_outputs):
            for i in range(self.n_inputs):
                self.synapses[j][i].eligibility = 0.0
                self.synapses[j][i].pre_timer = RSTDPSynapse.DISABLED
                self.synapses[j][i].post_timer = RSTDPSynapse.DISABLED