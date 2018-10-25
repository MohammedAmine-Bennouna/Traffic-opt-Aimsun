import threading
import time
import Connector as connector
import Metamodel as meta
import os
from pynput.keyboard import Controller, Key

if __name__ == "__main__":

    aim = connector.Connexion()
    aim.simulate()
    aim.interrupt()
