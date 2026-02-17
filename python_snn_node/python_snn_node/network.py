
import numpy as np
from .LIF import LIF

class LIFNetwork:
    def __init__(self, input_size, output_size, seed=None,
                 w_init_scale=0.1, wmin=0.01, wmax=1.0,
                 lif_kwargs=None):
        self.rng = np.random.default_rng(seed)
        self.input_size = int(input_size)
        self.output_size = int(output_size)

        lif_kwargs = lif_kwargs or {}
        # Pass w-grenser inn i neuronene (brukes i rSTDP/STDP clips)
        lif_kwargs.setdefault('wmin', wmin)
        lif_kwargs.setdefault('wmax', wmax)

        self.input_neurons = [LIF(**lif_kwargs) for _ in range(self.input_size)]
        self.output_neurons = [LIF(**lif_kwargs) for _ in range(self.output_size)]

        self.W = self.rng.uniform(0.0, w_init_scale, size=(self.input_size, self.output_size)).astype(float)
        self.wmin, self.wmax = float(wmin), float(wmax)

        # optional historikk
        self.mem_hist, self.thresh_hist, self.spike_hist, self.target_hist = [], [], [], []

        # standard dopamine-nivåer; kan overstyres i step()
        self.dopamine_correct = 1.0
        self.dopamine_wrong   = 0.5
        self.dopamine_nofire  = 0.1

    def step(self, input_values, correct_output,
             dopamine_correct=None, dopamine_wrong=None, dopamine_nofire=None):
        """
        input_values: liste/array med lengde == input_size (0..3 el. floats)
        correct_output: int i [0, output_size-1]
        return: (winner_idx, dopamine_used)
        """
        if len(input_values) != self.input_size:
            raise ValueError(f"Input length {len(input_values)} != input_size {self.input_size}")

        if dopamine_correct is not None: self.dopamine_correct = float(dopamine_correct)
        if dopamine_wrong   is not None: self.dopamine_wrong   = float(dopamine_wrong)
        if dopamine_nofire  is not None: self.dopamine_nofire  = float(dopamine_nofire)

        # 1) Oppdater input-nevroner med rå verdier (0..3 el. float)
        for i, val in enumerate(input_values):
            self.input_neurons[i].update(val)

        # 2) Beregn synaptisk input til output (bruk nåværende W)
        x = np.asarray(input_values, dtype=float)   # (I,)
        syn = x @ self.W                            # (O,)

        # 3) Oppdater output LIF
        for j in range(self.output_size):
            self.output_neurons[j].update(syn[j])

        # 4) Winner-take-all
        fired = [j for j, n in enumerate(self.output_neurons) if n.spk]
        if len(fired) > 0:
            winner_idx = int(self.rng.choice(fired))
            for j in range(self.output_size):
                if j != winner_idx:
                    self.output_neurons[j].spk = 0
        else:
            winner_idx = -1

        # 5) Dopamin
        if winner_idx == correct_output:
            dopamine = self.dopamine_correct
        elif winner_idx == -1:
            dopamine = self.dopamine_nofire
        else:
            dopamine = self.dopamine_wrong

        # 6) rSTDP - oppdater vekt etter at post er beregnet
        if winner_idx != -1:
            for i in range(self.input_size):
                pre_e = self.input_neurons[i].eligibility
                for j in range(self.output_size):
                    self.W[i, j] = self.output_neurons[j].rSTDP(
                        self.W[i, j], pre_e,
                        is_winner=(j == winner_idx),
                        dopamine=dopamine
                    )

        # 7) Logging (threshold feltet ditt heter 'threshold')
        self.mem_hist.append([n.mem for n in self.output_neurons])
        self.thresh_hist.append([n.threshold for n in self.output_neurons])
        self.spike_hist.append([n.spk for n in self.output_neurons])
        self.target_hist.append(int(correct_output))

        return winner_idx, dopamine
