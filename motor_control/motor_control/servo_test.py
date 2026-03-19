#!/usr/bin/python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import atexit

# Lokale Emakefun-filer (samme mappe)
from .Emakefun_MotorHAT import Emakefun_MotorHAT, Emakefun_DCMotor, Emakefun_Servo
from .Emakefun_MotorDriver import PWM
from .Emakefun_I2C import Emakefun_I2C
import time
mh = Emakefun_MotorHAT(addr=0x60)

myServo = mh.getServo(1)
while (True):

    myServo.writeServo(180)
    time.sleep(2)
    myServo.writeServo(0)
    time.sleep(2)