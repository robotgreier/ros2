import numpy as np
import csv
import os

# ==============================================================================
# SNN CORE CLASSES (Configured with User Parameters)
# ==============================================================================

class Layer:
    def __init__(self, n_inputs, n_outputs, neuron_params, synapse_params):
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        
        # Neuron parameters (updated to your specific request)
        self.decay     = neuron_params['decay']
        self.threshold = neuron_params['threshold']
        self.reset_val = neuron_params['reset']
        self.refractory = neuron_params['refractory']

        # Neuron state
        self.mem              = np.zeros(n_outputs, dtype=np.int32)
        self.pre_reset_mem    = np.zeros(n_outputs, dtype=np.int32)
        self.refractory_timer = np.zeros(n_outputs, dtype=np.int32)

        # Synapse parameters
        self.lr_shift    = synapse_params['lr_shift']
        self.t_pre       = synapse_params['t_pre']
        self.t_post      = synapse_params['t_post']
        self.tau_e_shift = synapse_params['tau_e_shift']
        self.dw_pos      = synapse_params['dw_pos']
        self.dw_neg      = synapse_params['dw_neg']

        # Initialize weights randomly (64-192) for a good test range
        self.weights     = np.random.randint(64, 192, size=(n_outputs, n_inputs), dtype=np.int32)
        self.eligibility = np.zeros((n_outputs, n_inputs), dtype=np.int32)
        self.pre_timer   = np.full((n_outputs, n_inputs), -1, dtype=np.int32)
        self.post_timer  = np.full((n_outputs, n_inputs), -1, dtype=np.int32)

    def forward(self, input_spikes):
        input_arr = np.asarray(input_spikes, dtype=np.int32)
        
        # LIF Dynamics
        synaptic_inputs = self.weights @ input_arr
        in_refractory = self.refractory_timer > 0
        
        # Integrated potential: mem = max(0, mem - decay) + inputs
        integrated = np.maximum(0, self.mem - self.decay) + synaptic_inputs
        self.mem = np.where(in_refractory, self.reset_val, integrated)
        self.pre_reset_mem = self.mem.copy()
        
        # Spiking
        output_arr = ((self.mem >= self.threshold) & ~in_refractory).astype(np.int32)
        self.mem = np.where(output_arr, self.reset_val, self.mem)
        self.refractory_timer = np.where(output_arr, self.refractory, np.maximum(0, self.refractory_timer - 1))

        # Eligibility update logic
        self._update_eligibility(input_arr, output_arr)
        return output_arr

    def _update_eligibility(self, pre_vec, post_vec):
        pre_mat = np.broadcast_to(pre_vec[np.newaxis, :], (self.n_outputs, self.n_inputs))
        post_mat = np.broadcast_to(post_vec[:, np.newaxis], (self.n_outputs, self.n_inputs))
        
        # Advance timers
        np.add(self.pre_timer, 1, out=self.pre_timer, where=self.pre_timer >= 0)
        np.add(self.post_timer, 1, out=self.post_timer, where=self.post_timer >= 0)
        self.pre_timer[self.pre_timer > self.t_pre] = -1
        self.post_timer[self.post_timer > self.t_post] = -1

        # STDP Pairings
        self.eligibility -= np.where(pre_mat.astype(bool) & (self.post_timer >= 0), self.dw_neg, 0)
        self.pre_timer[pre_mat.astype(bool)] = 0
        self.eligibility += np.where(post_mat.astype(bool) & (self.pre_timer >= 0), self.dw_pos, 0)
        self.post_timer[post_mat.astype(bool)] = 0

        # Trace Decay
        decay_amt = self.eligibility >> self.tau_e_shift
        decay_amt[(self.eligibility > 0) & (decay_amt == 0)] = 1
        self.eligibility -= decay_amt
        np.clip(self.eligibility, -256, 256, out=self.eligibility)

# ==============================================================================
# DATA GENERATION & EXECUTION
# ==============================================================================

SPIKE_SEQUENCE = [
    "101111110000101101011100111", "110010010101001111101110011", "100111110101110111011010000", "010101101011011000000100010",
    "001001001001000100101110010", "010001001101101111011111110", "110011010000110100110000100", "000100110010000000000011100",
    "011101110101100100000110010", "101000001101101010111011101", "011100110100000001100001111", "011000100000100110101111101",
    "010011110100011000001100011", "010100011100011111100001100", "000011000011011110111110101", "001010000100000010000000010",
    "001101110001111110011100101", "111101101001001010111111010", "101011011001001010111111100", "011000001010111001101010111",
    "001011010001111111010000101", "111000000001111100011111100", "011001001100000000010101001", "111000111011001000000000111",
    "110100100011001111111011001", "101101001010011100111101101", "101000110001100100101110010", "000101000110001001111011100",
    "001111011111100011001011000", "010111000000000111000101101", "001101001101110101111001100", "101100111110001101100001110",
    "001010101000000101110001001", "001110111000011010101000000", "111001110010100111111100101", "001111101000001100101100001",
    "101110111011010001011010111", "011010111010000000000110001", "101101111011110111110011010", "011000000110110000111001001",
    "001101001100010101111110110", "110010001010001011011101100", "000010101001001111101100001", "010110100001100111110111101",
    "000010111101001001101111010", "000110111011010110110000101", "010010001000110001100011111", "110100100010000000011011110",
    "100011001010111011001000000", "011100110011011000010111101", "110101001001000010100010000", "001010110111000101100000001",
    "011011001001111000111110100", "101010011010100000011110010", "110101010011011110010100001", "100111101001000110010000000",
    "010001000110111100111111010", "001101010101111000010101001", "101010010001001111001001100", "100001000011011011100011110",
    "000111111010011001110101010", "101010001110111111111101001", "011011000111011011111000110", "001010010111011110100101001",
    "010100010011001111100110100", "110001010100011011010111010", "011100001001010000100100100", "100001000001110100000110011",
    "111101100011011100110001000", "100001101101000001100000110", "111101110110010011010100011", "000000010110101101100111010",
    "111011100010111000010001101", "010101010010010110011001010", "000000010101001000000010111", "010001010001110110100110110",
    "011000001100011010001111111", "001001100001110010000011011", "000111011011110011010110010", "101011101101101001001000111",
    "011001100111001101011100110", "110001010101001111000111000", "101000001110010100010101110", "110000000111011100000111000",
    "100110101100111000000101111", "001100111101010001001010000", "110011110010110010100110000", "101000010000000001000111001",
    "111110010100001101010011010", "011001001000001000101100010", "001101110000001100101010001", "100111111001011001011000000",
    "110010011110110100000100011", "110010111001000100111110000", "000011010100110001110001001", "101100001100000110110111111",
    "011100101000001110011100101", "111111011100111100011100101", "111001100001110010000101111", "000010110101001101010110100",
]

def generate_files():
    np.random.seed(42)
    n_in, n_out = 27, 4
    
    n_params = {"decay": 600, "threshold": 1000, "reset": 0, "refractory": 1}
    s_params = {"lr_shift": 5, "t_pre": 3, "t_post": 1, "tau_e_shift": 2, "dw_pos": 10, "dw_neg": 8}
    
    net = Layer(n_in, n_out, n_params, s_params)

    # 1. Sparse Input Logic (First 12 bits reduced)
    sparse_sequence = []
    with open("input_stimulus.mem", "w") as f_stim:
        for bits in SPIKE_SEQUENCE:
            bit_list = list(bits) 
            # Thin out bits 0-11 (last 12 chars of the MSB string)
            for i in range(15, 27):
                if np.random.rand() > 0.15: bit_list[i] = "0"
            s = "".join(bit_list)
            sparse_sequence.append(s)
            f_stim.write(f"{int(s, 2):07X}\n")

    # 2. Initial Weights
    with open("weights_init.mem", "w") as f_w:
        for row in net.weights:
            for val in row:
                f_w.write(f"{int(val) & 0xFF:02X}\n")

    # 3. CSV Trace for bit-level verification
    header = ["t", "stim_hex"] + [f"mem{i}" for i in range(n_out)] + [f"spk{i}" for i in range(n_out)]
    with open("sparse_trace.csv", "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(header)
        for t, bits in enumerate(sparse_sequence):
            x = np.array([int(b) for b in reversed(bits)], dtype=np.int32)
            out = net.forward(x)
            row = [t, f"{int(bits, 2):07X}"] + net.pre_reset_mem.tolist() + out.tolist()
            writer.writerow(row)

    print("Success: generated 'input_stimulus.mem', 'weights_init.mem', and 'sparse_trace.csv'")

if __name__ == "__main__":
    generate_files()