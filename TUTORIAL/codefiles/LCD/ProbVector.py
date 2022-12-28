import struct
import numpy as np
import math

from pynq import MMIO
mmio = MMIO(0xB0030000,0x1FFF)

def DummyProbability(TVMOP):
    a = struct.pack('f',str(float(TVMOP*100)))
    balancedprob= 100-(TVMOP*100)
    b = struct.pack('f',str(float(balancedprob/2)))
    c = struct.pack('f',str(float(balancedprob/4)))
    d = struct.pack('f',str(float(balancedprob/8)))
    e = struct.pack('f',str(float(balancedprob/8)))
    mmio.write_mm(4,a)
    mmio.write_mm(8,b)
    mmio.write_mm(12,c)
    mmio.write_mm(16,d)
    mmio.write_mm(20,e)
    print('Probability of Cardiovascular complication is : ', a +'%')
    balancedprob= 100-(TVMOP*100)
    print('Probability of Respiratory complication is :', b +'%')
    print('Probability of Neurological complication is :', c +'%')
    print('Probability of Abdomen complication is :', d +'%')
    print('Probability of Infection complication is :', e +'%')
    
#    print('Probability of Cardiovascular complication is : ', str(float(TVMOP*100)) +'%')
#    balancedprob= 100-(TVMOP*100)
#    print('Probability of Respiratory complication is :', str(float(balancedprob/2)) +'%')
#    print('Probability of Neurological complication is :', str(float(balancedprob/4)) +'%')
#    print('Probability of Abdomen complication is :', str(float(balancedprob/8)) +'%')
#    print('Probability of Infection complication is :', str(float(balancedprob/8)) +'%')

#print(mmio.read(4))
#print(mmio.read(8))
#print(mmio.read(12))
#print(mmio.read(16))
#print(mmio.read(20))

