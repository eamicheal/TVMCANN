#  GNU nano 2.9.3
#  lcd.py                                  

import struct

from pynq import MMIO
mmio = MMIO(0xB0030000,0x1FFF)


a = struct.pack('f',0.7154)
b = struct.pack('f',0.0121)
c = struct.pack('f',0.1435)
d = struct.pack('f',0.1915)
e = struct.pack('f',0.0056)

mmio.write_mm(4,a)
mmio.write_mm(8,b)
mmio.write_mm(12,c)
mmio.write_mm(16,d)
mmio.write_mm(20,e)

#print(mmio.read(4))
#print(mmio.read(8))
#print(mmio.read(12))
#print(mmio.read(16))
#print(mmio.read(20))

