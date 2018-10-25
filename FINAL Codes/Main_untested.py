# -*- coding: utf-8 -*-
"""


"""

import threading
import time
import Connector as connector
import Métamodèle as meta
import os

def metamodelLauncher():
    
    aim = connector.Connexion()

    file = 'network.sqlite'
    file2 = 'complete.sqlite'
    path = "D:\2A\PSC\Code_Python"
    
    print("Importing network parameters... \n")
    
    p = meta.Aimsun_p()
    q = meta.Aimsun_q()
    N = meta.Aimsun_N()
    M = meta.Aimsun_M()
    
    print("Nbr of intersections: {} \n".format(M))
    print("Nbr of lanes: {} \n".format(N))
    
    w0 = 1 # poids de la composante physique dans la régression quadratique
    rMin = 0.05 # ratio minimal d'une phase d'un cycle sémaphorique
    totDur = 90 # durée d'un cycle de feux
    
    x0 = meta.random(p)
    xf = meta.optimise(x0, p, q, 5, 10, aim)
    
    aim.interrupt()
    

def aimsunLauncher():
    
    time.sleep(3)
    os.system("call D:\2A\PSC\Code_Python\Launch.bat")
    
    
    
if __name__ == "__main__":
    
    metamodelThread = threading.Thread(target = metamodelLauncher)
    aimsunThread = threading.Thread(target = aimsunLauncher)
    
    metamodelThread.start()
    aimsunThread.start()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

