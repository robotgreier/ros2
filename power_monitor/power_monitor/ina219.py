import smbus
import time

class INA219:
    REG_CALIB = 0x05
    REG_BUSV  = 0x02
    REG_SHUNTV = 0x01
    REG_POWER = 0x03
    REG_CURRENT = 0x04

    def __init__(self, address, bus=1):
        self.bus = smbus.SMBus(bus)
        self.addr = address

        # Calibration SEN0219 / INA219, 3.2A max, 0.1 ohm shunt
        calibration_value = 4096
        self.bus.write_word_data(self.addr, self.REG_CALIB, calibration_value)
        time.sleep(0.01)

        # Conversion factors
        self.current_lsb = 0.0001                   # 0.1 mA per bit
        self.power_lsb   = self.current_lsb * 20    # 20 mW per bit

    def read_voltage(self):
        raw = self.bus.read_word_data(self.addr, self.REG_BUSV)
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        return (raw >> 3) * 0.004     # 4mV per bit

    def read_current(self):
        raw = self.bus.read_word_data(self.addr, self.REG_CURRENT)
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        if raw & 0x8000:    # negative value
            raw -= 65536    # convert to signed
        return raw * self.current_lsb

    def read_power(self):
        raw = self.bus.read_word_data(self.addr, self.REG_POWER)
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        if raw & 0x8000:    # negative value
            raw -= 65536    # convert to signed
        return raw * self.power_lsb
