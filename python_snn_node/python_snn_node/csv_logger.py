import csv
import queue
import threading
from datetime import datetime

import os

"""
CSV LOGGING MODES

The SNN node supports three logging modes controlled by the ROS2 parameter:
    log_mode := "A" | "B" | "C"

Mode A — Input-triggered logging (Input → Next Output pairing)
---------------------------------------------------------------
• One CSV row per NEW input message.
• The input vector is paired with the NEXT network output
  produced by the timer loop.
• Best for supervised-style analysis:
    "Given this input, what did the network do?"

Mode B — Timer-based logging (Fixed rate, e.g., 30 Hz)
-------------------------------------------------------
• One CSV row per timer tick.
• Logs the latest input state and the current output.
• Produces steady-rate data (e.g., 30 rows/sec).
• Best for time-series analysis and plotting behavior over time.

Mode C — Event-based logging (Change detection)
-----------------------------------------------
• Logs only when something changes.
• Default change criteria:
    - Input vector changes OR
    - Winner neuron changes OR
    - Decision string changes
• Produces compact datasets.
• Best for event-driven / sparse SNN analysis.

All modes log:
    t_input_ns
    input_0 ... input_15
    t_output_ns
    winner
    decision
    spikes_0 ... spikes_N

CSV files are written asynchronously (non-blocking) to:
    ~/.ros/snn_logs/
One file is created per run with timestamped filename.
"""

class CsvAsyncLogger:
    def __init__(self, filepath: str, header: list[str], queue_size: int = 5000, flush_hz: float = 10.0):
        self.filepath = filepath
        self.header = header
        self.q = queue.Queue(maxsize=queue_size)
        self.flush_hz = max(flush_hz, 0.1)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped = 0

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._f = open(filepath, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.header)
        self._f.flush()

        self._thread.start()

    def push(self, row: list):
        try:
            self.q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        # Flush every 1/flush_hz seconds
        period = 1.0 / self.flush_hz
        next_flush = datetime.now().timestamp() + period

        while not self._stop.is_set():
            try:
                row = self.q.get(timeout=0.1)
                self._w.writerow(row)
            except queue.Empty:
                pass

            now = datetime.now().timestamp()
            if now >= next_flush:
                self._f.flush()
                next_flush = now + period

        # Final drain + close
        while True:
            try:
                row = self.q.get_nowait()
                self._w.writerow(row)
            except queue.Empty:
                break
        self._f.flush()
        self._f.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)