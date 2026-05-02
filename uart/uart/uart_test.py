import serial
import time
import sys

# Configuration based on main.sv and system setup[cite: 7]
PORT = '/dev/ttyAMA3'
BAUD = 250000 
SOF = 0xFF     

# Command Definitions from master.sv[cite: 8]
CMD_INIT = 0
CMD_SPIKE = 1
CMD_RESET = 4

def fletcher_checksum(data):
    """Calculates Fletcher-16 checksum used by verifier.sv[cite: 6, 12]"""
    sum_1 = 0
    sum_2 = 0
    for i, byte in enumerate(data):
        sum_1 = (sum_1 + byte) % 255
        sum_2 = (sum_2 + sum_1) % 255
    return sum_1, sum_2

def build_packet(cmd, payload, cmd_name=""):
    """Constructs and prints the raw packet structure[cite: 9]"""
    length = len(payload)
    header_and_data = [SOF, cmd, length] + payload
    s1, s2 = fletcher_checksum(header_and_data)
    full_packet = header_and_data + [s1, s2]
    
    print(f"\n[PACKET BUILDER: {cmd_name}]")
    print(f"  Header  : SOF={SOF} (0x{SOF:02X}), CMD={cmd}, LEN={length}")
    print(f"  Payload : {payload if len(payload) < 10 else f'{len(payload)} bytes of data'}")
    print(f"  Checksum: Sum1={s1}, Sum2={s2}")
    print(f"  Raw Hex : {' '.join(f'{b:02X}' for b in full_packet)}")
    
    return bytes(full_packet)

def run_verbose_test():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        print(f"--- Successfully opened {PORT} at {BAUD} baud ---")
    except Exception as e:
        print(f"FATAL: Could not open serial port: {e}")
        return

    # Step 1: System Reset
    # Triggering master_reset in master.sv[cite: 8]
    reset_pkt = build_packet(CMD_RESET, [], "RESET")
    print("Sending Reset...")
    ser.write(reset_pkt)
    time.sleep(0.1)

    # Step 2: Initialize Weights
    # master.sv expects exactly 128 bytes to exit INIT_LOOP[cite: 8]
    weights = [0x7F] * 128 
    init_pkt = build_packet(CMD_INIT, weights, "INIT_WEIGHTS")
    print(f"Sending {len(weights)} bytes of weight data...")
    ser.write(init_pkt)
    ser.flush()
    time.sleep(0.5)

    # Step 3: Send Spike Train
    # master.sv expects 11 bytes to trigger SPIKE_WRITE[cite: 8]
    spikes = [0xAA] * 11
    spike_pkt = build_packet(CMD_SPIKE, spikes, "SEND_SPIKES")
    print("Sending spike payload...")
    ser.write(spike_pkt)

    # Step 4: Verbose Listening Loop
    print("\n--- Entering Listening Mode (Press Ctrl+C to exit) ---")
    try:
        while True:
            if ser.in_waiting > 0:
                raw_rx = ser.read(ser.in_waiting)
                print(f"[{time.strftime('%H:%M:%S')}] RECEIVED {len(raw_rx)} bytes:")
                print(f"  HEX: {raw_rx.hex(' ').upper()}")
                # Basic check for SOF from packer.sv[cite: 9]
                if raw_rx[0] == 0xFF:
                    print("  Status: Valid Start of Frame detected.")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nTest terminated by user.")
    finally:
        ser.close()

if __name__ == "__main__":
    run_verbose_test()