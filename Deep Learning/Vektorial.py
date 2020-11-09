# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:00:02 2020

@author: lenovoz
"""
#anlasırdığı uzere vektörleştirmek for döngüsünden neredeyse 1000 kat daha hızlı
#(benim pc için) bu yuzden olabilfdiğince ANN(YSA) yaparken 4 döngüden kaçınmamz gerekir
import numpy as np
import time

a=np.array([1,2,3,4])
a=np.random.rand(1000000)   #random.rand kaca kac boyutlu bir matris oluşturuyor
b=np.random.rand(1000000)   #degerleride 0 ile birarsında belirliyor

timeS = time.time()
c1 = np.dot(a,b)
timeF = time.time()
print("Vectorized Version:",str(1000*(timeF-timeS))+" ms")

c = 0
timeS = time.time()
for i in range(1000000):
    c += a[i]*b[i]
timeF = time.time()
print("For Loop Version:",str(1000*(timeF-timeS))+" ms")