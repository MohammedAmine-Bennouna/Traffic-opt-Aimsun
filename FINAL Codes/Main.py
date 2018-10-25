import threading
import time
import Connector as connector
import Metamodel as meta
import os
from pynput.keyboard import Controller, Key
import matplotlib.pyplot as plt

file = 'network.sqlite'
file2 = 'complete.sqlite'
path = "D:/2A/PSC/Code_Python_v3/"

w0 = 1 # poids de la composante physique dans la régression quadratique
rMin = 0.05 # ratio minimal d'une phase d'un cycle sémaphorique
totDur = 90 # durée d'un cycle de feux

def metamodelLauncher():
    
    aim = connector.Connexion()
    
    print("Importing network parameters... \n")
    
    p = meta.Aimsun_p()
    q = meta.Aimsun_q()
    N = meta.Aimsun_N()
    M = meta.Aimsun_M()
    
    print("Nbr of intersections: {} \n".format(M))
    print("Nbr of lanes: {} \n".format(N))
    
    x0 = meta.random(p)
    listSet, listRes = meta.optimise(x0, p, q, 10, 20, aim)
    
    aim.interrupt()
    
    Y = listSet
    Z = listRes
    m = len(listSet)
    n = len(listRes)
    
    X = [i for i in range(m)]
    W = [m+i for i in range(n)]
    
    plt.plot(X,Y,'r.')
    plt.plot(W,Z,'b.')
    plt.xlabel("Itération")
    plt.ylabel("Temps de trajet total (h)")
    plt.grid()
    plt.show()


def aimsunLauncher():
    
    time.sleep(3)
    os.system(path + 'Launch.bat')    

def enter():
    
    time.sleep(5)
    keyboard = Controller()
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)


if __name__ == "__main__":

    metamodelThread = threading.Thread(target = metamodelLauncher)
    aimsunThread = threading.Thread(target = aimsunLauncher)
    enterKey = threading.Thread(target = enter)
    
    metamodelThread.start()
    aimsunThread.start()
    enterKey.start()
