# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:21:44 2016

@author: Yassine
"""
import socket, json, random, os

""" 

"""

class Connexion:
    socket = None
    connection = None
    adress = None
    
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('localhost', 2199))
        self.socket.listen(1)
        self.connection, self.address = self.socket.accept()
        print("Aimsun connection successfully initialized.")

    def interrupt(self):
        request = json.dumps(["INTERRUPT"]).encode()
        self.connection.send(request)
        buf = self.connection.recv(4).decode()
        print("Connection closed.")

    def setParams(self,params):
        request = json.dumps(["SET"] + params).encode()
        self.connection.send(request)
        buf = self.connection.recv(4).decode()
        return

    def simulate(self):
        request = json.dumps(["SIMULATE"]).encode()
        self.connection.send(request)
        buf = self.connection.recv(4).decode()
        return

