import serial, time

ser = serial.Serial('/dev/ttyUSB0', 57600)

print('Locking left...')
ser.write('steer, 40')
time.sleep(1)

print('Locking right...')
ser.write('steer, 140')
time.sleep(1)

print('motor forward')
ser.write('motor, 110')
time.sleep(1)

print('motor stop')
ser.write('motor, 90')

