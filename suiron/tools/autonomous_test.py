import serial, time

ser = serial.Serial('/dev/ttyUSB0', 57600)

time.sleep(2)

print('Locking left...')
ser.write('steer,40\n')
time.sleep(1)

print('Locking right...')
ser.write('steer,140\n')
time.sleep(1)

print('motor forward')
ser.write('steer,90\n')
time.sleep(0.02)
ser.write('motor,78\n')
time.sleep(5)

print('motor stop')
ser.write('motor,90\n')

