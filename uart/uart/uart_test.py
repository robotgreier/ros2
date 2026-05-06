import serial
import time

# --- Setup ---
PORT = '/dev/ttyAMA3'
BAUD = 250000 
SOF = 0xFF     

def send_packet(ser, cmd, payload):
    """Simple construction of [SOF][CMD][LEN][DATA][CSUM1][CSUM2][cite: 9, 12]"""
    pkt_body = [SOF, cmd, len(payload)] + payload
    s1 = s2 = 0
    for b in pkt_body:
        s1 = (s1 + b) % 255
        s2 = (s2 + s1) % 255
    
    full_pkt = bytes(pkt_body + [s1, s2])
    ser.write(full_pkt)
    return full_pkt.hex().upper()

def run_test():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    
    # 1. Hardware Reset
    print(f"Resetting... {send_packet(ser, 4, [])}")
    time.sleep(2) 

    # 2. Initialize Weights (128 bytes required)
    weights = [0x7F] * 128 
    print(f"Initializing Weights... {send_packet(ser, 0, weights)[:30]}...")
    time.sleep(2) 

    # 3. Send Spikes (11 bytes required)[cite: 8]
    # Stimulates neurons using 3-bit-per-byte doubling[cite: 3]
    spikes = [0x3F] + [0x00] * 10 
    print(f"Sending Spikes... {send_packet(ser, 1, spikes)}")

    print("\n--- Monitoring Output ---")
    try:
        while True:
            if ser.in_waiting >= 5: # Minimum packet size
                raw = ser.read(ser.in_waiting)
                print(f"RX: {raw.hex().upper()}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        ser.close()

if __name__ == "__main__":
    run_test()
