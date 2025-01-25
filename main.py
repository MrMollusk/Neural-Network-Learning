import pygame
import RPi.GPIO as GPIO
from time import sleep

# Setup GPIO pins (change these according to your motor driver connections)
BASE_MOTOR_DIR = 23  # Direction pin for base motor
BASE_MOTOR_PUL = 22  # Pulse pin for base motor
ARM_MOTOR_DIR = 27   # Direction pin for arm motor
ARM_MOTOR_PUL = 17   # Pulse pin for arm motor

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BASE_MOTOR_DIR, GPIO.OUT)
GPIO.setup(BASE_MOTOR_PUL, GPIO.OUT)
GPIO.setup(ARM_MOTOR_DIR, GPIO.OUT)
GPIO.setup(ARM_MOTOR_PUL, GPIO.OUT)

# Initialize pygame and Xbox controller
pygame.init()

# Initialize the Xbox controller
joystick = pygame.joystick.Joystick(0)
joystick.init()

def generate_pulses(pul_pin, pulse_count, pulse_speed):
    """Generate pulses for motor control."""
    for _ in range(pulse_count):
        GPIO.output(pul_pin, GPIO.HIGH)
        sleep(pulse_speed)
        GPIO.output(pul_pin, GPIO.LOW)
        sleep(pulse_speed)

def control_motor(x_input, dir_pin, pul_pin):
    speed = int(x_input * 100)  # Scale the joystick input to speed (100 steps max)
    print(f"Speed: {speed}")  # Debug: Print the speed based on joystick input
    if speed > 0:
        GPIO.output(dir_pin, GPIO.HIGH)  # Forward direction
        generate_pulses(pul_pin, speed, 0.01)  # Adjusted pulse speed for faster movement
    elif speed < 0:
        GPIO.output(dir_pin, GPIO.LOW)  # Reverse direction
        generate_pulses(pul_pin, -speed, 0.01)  # Adjusted pulse speed for faster movement
    else:
        GPIO.output(pul_pin, GPIO.LOW)  # Stop motor if speed is 0

# Main loop
try:
    while True:
        pygame.event.pump()

        # Get joystick axis values
        x_axis = joystick.get_axis(0)  # Left joystick X-axis (base motor)
        y_axis = joystick.get_axis(1)  # Right joystick Y-axis (arm motor)

        # Debug: Print the joystick axis values
        print(f"X-axis: {x_axis}, Y-axis: {y_axis}")

        # Control the motors based on joystick input
        control_motor(x_axis, BASE_MOTOR_DIR, BASE_MOTOR_PUL)
        control_motor(y_axis, ARM_MOTOR_DIR, ARM_MOTOR_PUL)

        sleep(0.01)  # Small delay to avoid overloading the processor

except KeyboardInterrupt:
    print("Exiting program")

finally:
    GPIO.cleanup()  # Ensure GPIO pins are cleaned up
    pygame.quit()
