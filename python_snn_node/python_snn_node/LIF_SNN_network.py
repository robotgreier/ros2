import numpy as np


class LIF:
    """
    Leaky Integrate-and-Fire neuron.

    Args:
        decay: Membrane decay (uint8, scaled x256)
        threshold: Spike threshold (uint16, scaled x256)
        reset: Membrane reset value after spike
    """

    def __init__(self, decay=256, threshold=1024, reset=0):
        self.decay = decay
        self.threshold = threshold
        self.reset = reset
        self.mem = 0            # uint16, never negative
        self.pre_reset_mem = 0  # Membrane potential before reset, used for WTA
        self.spk = 0

    def update(self, synaptic_input):
        """Membrane update. Returns spike (0 or 1)."""
        self.spk = 0
        self.mem = max(0, self.mem - self.decay) + synaptic_input
        self.pre_reset_mem = self.mem

        if self.mem >= self.threshold:
            self.spk = 1
            self.mem = self.reset

        return self.spk


class RSTDPSynapse:
    """
    Reward-modulated STDP synapse with rectangular window.

    Modes:
        'rstdp': Weight updates only when apply_reward() is called with a dopamine
                 signal. Eligibility trace accumulates STDP events and decays over time.
        'stdp':  Weight updates immediately on each spike pairing (dopamine=1).

    Args:
        mode: 'rstdp' (default) or 'stdp'
        lr_shift: Learning rate as right-shift (lr = 1 / 2^lr_shift)
        w_init: Initial weight (random if None)
        t_pre/t_post: Rectangular STDP window widths (timesteps)
        tau_e_shift: Eligibility decay as right-shift
        dw_pos/dw_neg: Fixed weight increment/decrement on spike pairing
        w_min/w_max: Weight clamps
    """

    DISABLED = -1

    def __init__(self, lr_shift=2, w_init=None,
                 t_pre=2, t_post=2, tau_e_shift=2,
                 dw_pos=16, dw_neg=64,
                 w_min=8, w_max=255,
                 mode='rstdp'):

        self.mode = mode
        self.lr_shift = lr_shift
        self.weight = w_init if w_init is not None else np.random.randint(64, 192)

        self.t_pre = t_pre
        self.t_post = t_post
        self.tau_e_shift = tau_e_shift
        self.dw_pos = dw_pos
        self.dw_neg = dw_neg

        self.pre_timer = self.DISABLED
        self.post_timer = self.DISABLED
        self.eligibility = 0  # int16, range [-256, 256]

        self.w_min = w_min
        self.w_max = w_max

    def update_eligibility(self, pre_spike, post_spike):
        """Update spike timing counters and eligibility each timestep."""
        # Advance and expire timers
        if self.pre_timer >= 0:
            self.pre_timer += 1
            if self.pre_timer > self.t_pre:
                self.pre_timer = self.DISABLED

        if self.post_timer >= 0:
            self.post_timer += 1
            if self.post_timer > self.t_post:
                self.post_timer = self.DISABLED

        # LTD: pre fires while post_timer is active (acausal pairing)
        if pre_spike:
            if self.post_timer >= 0:
                self.eligibility -= self.dw_neg
            self.pre_timer = 0

        # LTP: post fires while pre_timer is active (causal pairing)
        if post_spike:
            if self.pre_timer >= 0:
                self.eligibility += self.dw_pos
            self.post_timer = 0

        self.eligibility -= self.eligibility >> self.tau_e_shift
        self.eligibility = max(-256, min(256, self.eligibility))

        # In stdp mode, apply weight update immediately (dopamine=1 → ×1 multiplier)
        if self.mode == 'stdp':
            self.apply_reward(dopamine=1)

    def apply_reward(self, dopamine):
        """
        Apply reward-modulated weight update.

        HDL mapping:
            - dopamine=0 is a no-op
            - Signed multiply: product = eligibility * dopamine (one DSP48E1)
            - Arithmetic right-shift by lr_shift
            - Saturating add to weight, then clamp

        Args:
            dopamine: Signed integer. Positive = reward, negative = punishment.
                      Zero = no-op. Magnitude controls update strength.

        The signed multiply preserves both signals: eligibility sign carries STDP
        pairing direction (LTP vs LTD), dopamine sign carries reward vs punishment.
        """
        if dopamine == 0:
            return

        delta_w = (self.eligibility * dopamine) >> self.lr_shift
        new_weight = self.weight + delta_w
        self.weight = max(self.w_min, min(self.w_max, new_weight))


class SNNLayer:
    """
    Fully-connected SNN layer with vectorised NumPy state arrays.

    Args:
        n_inputs: Number of pre-synaptic input neurons
        n_outputs: Number of post-synaptic output neurons
        neuron_params: Dict passed to LIF constructor
        synapse_params: Dict passed to RSTDPSynapse constructor;
                        include 'mode': 'stdp' or 'rstdp' (default)
        feedback: If True, adds one extra input driven by NOR of previous outputs
    """

    def __init__(self, n_inputs, n_outputs, neuron_params=None, synapse_params=None, feedback=False):
        neuron_params  = neuron_params  or {}
        synapse_params = synapse_params or {}

        self.n_outputs = n_outputs
        self.mode      = synapse_params.get('mode', 'rstdp')
        self.feedback  = feedback
        self._feedback_reg = 0

        # +1 input for feedback neuron if enabled
        self.n_inputs = n_inputs + 1 if feedback else n_inputs

        # Neuron state (n_outputs,)
        self.decay     = neuron_params.get('decay',     192)
        self.threshold = neuron_params.get('threshold', 1024)
        self.reset_val = neuron_params.get('reset',     0)

        self.mem           = np.zeros(n_outputs, dtype=np.int32)
        self.pre_reset_mem = np.zeros(n_outputs, dtype=np.int32)
        self.spk           = np.zeros(n_outputs, dtype=np.int32)
        self.spike_count   = np.zeros(n_outputs, dtype=np.int32)

        # Synapse state (n_outputs, n_inputs)
        self.lr_shift    = synapse_params.get('lr_shift',    3)
        self.t_pre       = synapse_params.get('t_pre',       2)
        self.t_post      = synapse_params.get('t_post',      3)
        self.tau_e_shift = synapse_params.get('tau_e_shift', 4)
        self.dw_pos      = synapse_params.get('dw_pos',      64)
        self.dw_neg      = synapse_params.get('dw_neg',      8)
        self.w_min       = synapse_params.get('w_min',       8)
        self.w_max       = synapse_params.get('w_max',       255)

        w_init = synapse_params.get('w_init', None)
        if w_init is None:
            self.weights = np.random.randint(64, 192, size=(n_outputs, self.n_inputs), dtype=np.int32)
        else:
            self.weights = np.full((n_outputs, self.n_inputs), w_init, dtype=np.int32)

        self.eligibility = np.zeros((n_outputs, self.n_inputs), dtype=np.int32)
        self.pre_timer   = np.full((n_outputs, self.n_inputs), -1, dtype=np.int32)
        self.post_timer  = np.full((n_outputs, self.n_inputs), -1, dtype=np.int32)

    def forward(self, input_spikes):
        """
        Process one timestep.

        Args:
            input_spikes: Array of length n_inputs (0s and 1s)

        Returns:
            List of output spikes (length n_outputs)
        """
        input_arr = np.asarray(input_spikes, dtype=np.int32)
        if self.feedback:
            input_arr = np.append(input_arr, self._feedback_reg)

        # LIF membrane update
        synaptic_inputs = self.weights @ input_arr
        self.mem = np.maximum(0, self.mem - self.decay) + synaptic_inputs
        self.pre_reset_mem = self.mem.copy()
        output_arr = (self.mem >= self.threshold).astype(np.int32)
        self.mem = np.where(output_arr, self.reset_val, self.mem)
        self.spike_count += output_arr

        if self.mode == 'stdp':
            winner = self.winner_takes_all(output_arr)
            # Lateral inhibition: suppress all non-winner membranes
            self.mem[np.arange(self.n_outputs) != winner] = self.reset_val

            # Only winner's synapses see real spikes; losers decay only
            pre_mat  = np.zeros((self.n_outputs, self.n_inputs), dtype=np.int32)
            post_mat = np.zeros((self.n_outputs, self.n_inputs), dtype=np.int32)
            pre_mat[winner]  = input_arr
            post_mat[winner] = output_arr[winner]
            self._update_eligibility(pre_mat, post_mat)

            # Immediate weight update for all synapses (dopamine=1 → signed multiply ×1)
            delta_w = self.eligibility >> self.lr_shift
            self.weights = np.clip(self.weights + delta_w, self.w_min, self.w_max)
        else:
            pre_mat  = np.broadcast_to(input_arr[np.newaxis, :],  (self.n_outputs, self.n_inputs))
            post_mat = np.broadcast_to(output_arr[:, np.newaxis], (self.n_outputs, self.n_inputs))
            self._update_eligibility(pre_mat, post_mat)

        if self.feedback:
            self._feedback_reg = int(not np.any(output_arr))

        return output_arr.tolist()

    def _update_eligibility(self, pre_mat, post_mat):
        """Update all (n_outputs x n_inputs) eligibility traces for one timestep."""
        # Advance active timers
        np.add(self.pre_timer,  1, out=self.pre_timer,  where=self.pre_timer  >= 0)
        np.add(self.post_timer, 1, out=self.post_timer, where=self.post_timer >= 0)

        # Expire timers
        self.pre_timer[self.pre_timer   > self.t_pre]  = -1
        self.post_timer[self.post_timer > self.t_post] = -1

        # LTD: pre fires while post_timer is active (acausal)
        pre_fired = pre_mat.astype(bool)
        np.subtract(self.eligibility, self.dw_neg,
                    out=self.eligibility,
                    where=pre_fired & (self.post_timer >= 0))
        self.pre_timer[pre_fired] = 0

        # LTP: post fires while pre_timer is active (causal)
        # Uses updated pre_timer so simultaneous pre+post counts as LTP
        post_fired = post_mat.astype(bool)
        np.add(self.eligibility, self.dw_pos,
               out=self.eligibility,
               where=post_fired & (self.pre_timer >= 0))
        self.post_timer[post_fired] = 0

        # Decay and clamp
        self.eligibility -= self.eligibility >> self.tau_e_shift
        np.clip(self.eligibility, -256, 256, out=self.eligibility)

    def winner_takes_all(self, output_spikes):
        """
        Returns index of winning neuron.
        Spiking neurons beat non-spiking ones; pre-reset membrane breaks ties.
        """
        spikes = np.asarray(output_spikes, dtype=np.int32)
        spiking = np.where(spikes == 1)[0]
        # If any neurons spiked, compete among them; otherwise all compete
        pool = spiking if len(spiking) > 0 else np.arange(len(spikes))
        # Highest pre-reset membrane potential wins (deterministic, no index bias)
        return int(pool[np.argmax(self.pre_reset_mem[pool])])

    def apply_reward(self, dopamine, winner_idx):
        """
        Apply reward to the winning neuron's synapses.
        No-op in 'stdp' mode (weights already updated in forward()).

        Args:
            dopamine:    Signed integer. Positive = reward, negative = punishment.
                         Zero = no-op. Magnitude controls update strength.
            winner_idx:  Row index of the winning neuron.
        """
        if self.mode == 'stdp':
            return

        delta_w = (self.eligibility[winner_idx] * dopamine) >> self.lr_shift
        new_row = self.weights[winner_idx] + delta_w
        np.clip(new_row, self.w_min, self.w_max, out=self.weights[winner_idx])

    def get_weights(self):
        """Return weight matrix as numpy array [n_outputs x n_inputs]."""
        return self.weights.copy()

    def load_weights(self, weight_file="weights.mem"):
        """Load weights from a hex file (one value per line)."""
        with open(weight_file) as f:
            hex_values = [int(line.strip(), 16) for line in f]
        self.weights[:] = np.array(hex_values, dtype=np.int32).reshape(self.n_outputs, self.n_inputs)

    def reset_state(self):
        """Reset all neuron and synapse trace state."""
        self.mem[:]           = 0
        self.spk[:]           = 0
        self.pre_reset_mem[:] = 0
        self.eligibility[:]   = 0
        self.pre_timer[:]     = -1
        self.post_timer[:]    = -1
        self._feedback_reg    = 0
        self.spike_count[:]   = 0
