import serial
import time
import sys

# --- Configuration ---
PORT = '/dev/ttyAMA3'
BAUD = 250000 
SOF = 0xFF     

# Command Definitions from master.sv
CMD_INIT = 0
CMD_SPIKE = 1
CMD_RESET = 4

def fletcher_checksum(data):
    """Calculates Fletcher-16 checksum used by verifier.sv[cite: 6, 12]"""
    sum_1 = 0
    sum_2 = 0
    for byte in data:
        sum_1 = (sum_1 + byte) % 255
        sum_2 = (sum_2 + sum_1) % 255
    return sum_1, sum_2

def pack_spikes(spikes):
    """Matches spike_codec.py logic: [a,b,c] -> 00 aa bb cc[cite: 3]"""
    packed = []
    # FPGA expects 31 spikes total[cite: 7]
    for i in range(0, len(spikes), 3):
        a = spikes[i] if i < len(spikes) else 0
        b = spikes[i + 1] if i + 1 < len(spikes) else 0
        c = spikes[i + 2] if i + 2 < len(spikes) else 0

        byte = (
            (a << 5) | (a << 4) |
            (b << 3) | (b << 2) |
            (c << 1) | c
        )
        packed.append(byte)
    return packed

def build_packet(cmd, payload, cmd_name=""):
    """Constructs a raw packet: [SOF][CMD][LEN][DATA...][CSUM1][CSUM2]"""
    length = len(payload)
    header_and_data = [SOF, cmd, length] + payload
    s1, s2 = fletcher_checksum(header_and_data)
    full_packet = header_and_data + [s1, s2]
    
    print(f"\n[SENDING: {cmd_name}]")
    print(f"  Raw Hex : {' '.join(f'{b:02X}' for b in full_packet)}")
    return bytes(full_packet)

def read_full_packet(ser):
    """Synchronizes with SOF and reads a complete frame[cite: 9, 12]"""
    while ser.in_waiting > 0:
        byte = ser.read(1)
        if byte and byte[0] == SOF:
            # Read CMD and LEN bytes
            header = ser.read(2)
            if len(header) < 2: return None
            
            cmd = header[0]
            length = header[1]
            
            # Read Payload + 2 Checksum bytes[cite: 9]
            body = ser.read(length + 2)
            if len(body) < (length + 2): return None
            
            full_frame = bytes([SOF]) + header + body
            return {
                "cmd": cmd,
                "payload": body[:length],
                "checksum": body[length:],
                "raw": full_frame
            }
    return None

def run_test():
    try:
        # Use a short timeout for non-blocking reads
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        print(f"--- Connected to {PORT} ---")
    except Exception as e:
        print(f"FATAL: {e}")
        return

    # 1. Reset System[cite: 8]
    ser.write(build_packet(CMD_RESET, [], "RESET"))
    time.sleep(0.2) # Wait for master_reset logic to settle[cite: 8]

    # 2. Initialize Weights[cite: 8]
    # master.sv expects 128 bytes to fill the weight memory[cite: 8]
    weights = [0x7F] * 128 
    ser.write(build_packet(CMD_INIT, weights, "INIT_WEIGHTS"))
    
    # Critical: master.sv runs a loop for 128 cycles; give it time[cite: 8]
    time.sleep(0.5) 

    # 3. Send Packed Spikes[cite: 3, 8]
    # Stimulate the first 3 neurons (bits 1,1,1)
    raw_spikes = [1, 1, 1] + ([0] * 28) 
    spike_payload = pack_spikes(raw_spikes)
    ser.write(build_packet(CMD_SPIKE, spike_payload, "SEND_SPIKES"))

    print("\n--- Listening for Response ---")
    try:
        while True:
            packet = read_full_packet(ser)
            if packet:
                print(f"\n[{time.strftime('%H:%M:%S')}] RECEIVED PACKET:")
                print(f"  CMD: {packet['cmd']} (0=Spikes, 2=Error)[cite: 8]")
                print(f"  DATA: {packet['payload'].hex(' ').upper()}")
                print(f"  CSUM: {packet['checksum'].hex(' ').upper()}")
                
                # Check if this is a spike output packet[cite: 4, 8]
                if packet['cmd'] == 0 and len(packet['payload']) > 0:
                    byte = packet['payload'][0]
                    # FPGA output is one-hot[cite: 3]
                    fired = [i for i in range(4) if (byte >> i) & 1]
                    print(f"  Neurons Fired: {fired if fired else 'None'}")
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nTest Closed.")
    finally:
        ser.close()

if __name__ == "__main__":
    run_test()