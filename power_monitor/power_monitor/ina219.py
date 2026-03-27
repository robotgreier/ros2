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

        # Calibration INA219, 6A max, 0.1 ohm shunt
        calibration_value = 2238
        self.bus.write_word_data(self.addr, self.REG_CALIB, calibration_value)

        time.sleep(0.01)

        # Conversion factors
        self.current_lsb = 0.000183      # 0.183 mA per bit
        self.power_lsb   = self.current_lsb * 20

    def read_voltage(self):
        raw = self.bus.read_word_data(self.addr, self.REG_BUSV)
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        return (raw >> 3) * 0.004     # 4mV per bit

    def read_current(self):
        raw = self.bus.read_word_data(self.addr, self.REG_CURRENT)
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        return raw * self.current_lsb

    def read_power(self):
        raw = self.bus.read_word_data(self.addr, self.REG_POWER)
        raw = ((raw & 0xFF) << 8) | (raw >> 8)
        return raw * self.power_lsb
