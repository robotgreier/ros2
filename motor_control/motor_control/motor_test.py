#!/usr/bin/env python3
import time
import atexit

from Emakefun_MotorHAT import Emakefun_MotorHAT

# ---------------------------------------------------------
# Motoroppsett
# ---------------------------------------------------------
mh = Emakefun_MotorHAT(addr=0x60)

left  = mh.getMotor(1)   # M1
right = mh.getMotor(2)   # M2

def turn_off():
    left.run(Emakefun_MotorHAT.RELEASE)
    right.run(Emakefun_MotorHAT.RELEASE)

atexit.register(turn_off)

# ---------------------------------------------------------
# Hjelpefunksjoner
# ---------------------------------------------------------
def drive(motor, speed, name):
    print(f">>> {name} speed={speed}")
    if speed == 0:
        motor.run(Emakefun_MotorHAT.RELEASE)
        motor.setSpeed(0)
        return

    if speed > 0:
        motor.run(Emakefun_MotorHAT.FORWARD)
    else:
        motor.run(Emakefun_MotorHAT.BACKWARD)

    motor.setSpeed(abs(speed))


def wait():
    time.sleep(1.5)
    turn_off()
    time.sleep(0.5)

# ---------------------------------------------------------
# TESTSEKVENS
# ---------------------------------------------------------
print("\n======= MOTOR TEST STARTER =======\n")

print("1) Venstre motor forover")
drive(left, 150, "Venstre FORWARD")
wait()

print("2) Venstre motor bakover")
drive(left, -150, "Venstre BACKWARD")
wait()

print("3) Høyre motor forover")
drive(right, 150, "Høyre FORWARD")
wait()

print("4) Høyre motor bakover")
drive(right, -150, "Høyre BACKWARD")
wait()

print("5) Begge forover")
drive(left, 150, "Venstre FORWARD")
drive(right, 150, "Høyre FORWARD")
wait()

print("6) Venstresving (venstre bak, høyre frem)")
drive(left, -150, "Venstre BACKWARD")
drive(right, 150, "Høyre FORWARD")
wait()

print("7) Høyresving (venstre frem, høyre bak)")
drive(left, 150, "Venstre FORWARD")
drive(right, -150, "Høyre BACKWARD")
wait()

print("\n======= MOTOR TEST FULLFØRT =======\n")
turn_off()
