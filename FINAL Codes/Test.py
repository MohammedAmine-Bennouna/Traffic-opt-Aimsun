# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:56:28 2018

@author: verne
"""

import sys, socket, json
import time
from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *
from PyMesoPlugin import *


""" 
This class implements the interface between Python and Aimsun. It allows us to directly call two important
functions : 
    -runSimulation, which calls for a simulation to be run.
    -setParams, which modifies the parameters inside the simulation model.
"""

import Param_feux as prf



class Communicator():
    console = None
    model = None
    replication = None
    simulator = None
    plugin = None

    def __init__(self, path, replication_id):
        self.console = ANGConsole()
        
        # Load a network
        if( not self.console.open(path) ):
            print("Error loading file.")
            return
        print("File opened.")
        
        self.model = self.console.getModel()
        self.plugin = GKSystem.getSystem().getPlugin( "GGetram" )
        self.simulator = self.plugin.getCreateSimulator( self.model )
        self.replication = self.model.getCatalog().find(replication_id)
        
        return
    
    def runSimulation(self):
        self.simulator.addSimulationTask(GKSimulationTask(self.replication,GKReplication.eBatch))
        print("simulator instantiated")
        self.simulator.simulate()
        print("simulation launched")
        return

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
        

######################################

class ListeningThread():
    socket = None
    comm = None
    
    def __init__(self, communicator):
#        Thread.__init__(self)        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', 2199))
        self.comm = communicator       
        print("Aimsun connection successfully initialized.")
        return


    def run(self):
        while True:
            print("\n Waiting for request.......")
            buff = self.socket.recv(4096)
            rawData = json.loads(buff)
            requestType = rawData[0]
            if requestType == "SIMULATE":
                time.sleep(3)
                self.comm.runSimulation()
                self.comm.plugin.readResult( self.comm.replication )
                self.socket.send("Simulation done!")
                self.socket.send("1")
            if requestType == "SET":
                print("SET.")                
                params = rawData[1:]
                self.comm.setParams(params)
                self.socket.send("Parameters set.")
                self.socket.send("1")
            if requestType == "INTERRUPT":
                print("INTERRUPT.")
                break
                self.socket.send("The term signal has been received. System shutdown!")
                self.socket.send("1")
        return



def main(argv):
    path = argv[1]
    replication = int(argv[2])
    communicator = Communicator(path, replication)
    communicator.runSimulation()
    
if __name__ == "__main__":
    sys.exit(main(sys.argv)) 