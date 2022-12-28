
import numpy as np
import math

def DummyProbability(TVMOP):
    print('Probability of Cardiovascular complication is : ', str(float(TVMOP*100)) +'%')
    balancedprob= 100-(TVMOP*100)
    print('Probability of Respiratory complication is :', str(float(balancedprob/2)) +'%')
    print('Probability of Neurological complication is :', str(float(balancedprob/4)) +'%')
    print('Probability of Abdomen complication is :', str(float(balancedprob/8)) +'%')
    print('Probability of Infection complication is :', str(float(balancedprob/8)) +'%')


