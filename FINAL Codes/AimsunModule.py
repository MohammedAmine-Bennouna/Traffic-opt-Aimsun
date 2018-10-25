# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 14:58:57 2016

@author: Yassine
"""

import sys, socket, json
from threading import Thread
from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *
from PyMesoPlugin import *
import Param_feux as prf


class ListeningThread(Thread):
    socket = None
    aimsunInterface = None

    def __init__(self, aimsunInterface):
        Thread.__init__(self)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', 2199))
        self.aimsunInterface = aimsunInterface
        return

    def run(self):
        while True:
            buff = self.socket.recv(4096)
            rawData = json.loads(buff)
            requestType = rawData[0]
            if requestType == "SIMULATE":
                self.aimsunInterface.runSimulation()
            if requestType == "SET":
                params = rawData[1:]
                self.aimsunInterface.setParams(params)
            if requestType == "INTERRUPT":
                break
            self.socket.send("DONE")
            print('Request executed.')
        return


""" 
This class implements the interface between Python and Aimsun. It allows us to directly call two important
functions : 
    -runSimulation, which calls for a simulation to be run.
    -setParams, which modifies the parameters inside the simulation model.
"""

class AimsunInterface:
    console = None
    model = None
    replication = None
    simulator = None
    plugin = None

    def __init__(self, path, replication):
        self.console = ANGConsole()
        # Load a network
        if not self.console.open(path):
            print("Error loading file.")
            return
        print("File opened.")
        self.model = self.console.getModel()
        self.replication = self.model.getCatalog().find(replication)
        self.plugin = GKSystem.getSystem().getPlugin( "AMesoPlugin" )
        self.simulator = self.plugin.getCreateSimulator( self.model )
        return

    def runSimulation(self):

        self.simulator.addSimulationTask( GKSimulationTask(self.replication,GKReplication.eBatch) )
        print("Simulation task added.")
        self.simulator.simulate()
        print("Simulation done.")

        

    def setParams(self, params):
        
        print("Setting params...")        
        
        for param in params:
    
            id_n = param[0]
            id_CP = param[1]
            phases = param[2] 
    
            prf.phaseChange(self.model, id_n, id_CP, phases)           
            
        self.console.save("network.ang")
        print("Control Plan modified.")

        return

def main(argv):
    path = argv[1]
    replication = int(argv[2])
    aimInter = AimsunInterface(path, replication)
    listeningThread = ListeningThread(aimInter)
    listeningThread.start()

if __name__ == "__main__":
     sys.exit(main(sys.argv)) 