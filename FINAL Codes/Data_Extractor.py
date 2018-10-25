# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:49:24 2018
@author: yassine
"""

import numpy as np
import sqlite3


file = 'network.sqlite'

def getTotalTravelTime(database):
    """renvoie le temps de trajet total pour la simulation database"""
    
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("select traveltime from MISYS where ent == 0 and sid == 0")
    T = cursor.fetchall()[0][0]
    cursor.execute("update MISYS set traveltime = 0 where ent == 0 and sid == 0")
    conn.commit()
    conn.close()
    
    return T

def getLanesNumber(database):
    """renvoie le nombre de files pour la simulation database
    database : string contenant le nom du fichier sqlite"""
    
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("select COUNT() as NbFiles from ( select DISTINCT oid, lane from MILANE )")
    N = cursor.fetchall()[0][0]
    conn.commit()
    conn.close()
    
    return N

def getNetworkParameters(database):
    """renvoie les paramètres du réseau pour la simulation database
    database : string contenant le nom du fichier sqlite
    sortie :  tuple of arrays (nb of nodes, [id_CP, id_n, avail. cyc. ratio, nb of phases, start index] )"""
    
    conn = sqlite3.connect(database)
    
    cursor = conn.cursor()
    cursor.execute("SELECT count(DISTINCT node_id) as nb_node FROM CONTROLPHASE")
    M = cursor.fetchall()[0][0]
    cursor.execute("SELECT oid as CP_id, node_id, sum(active_time_percentage) as avCycRat, count() as nbPhases FROM CONTROLPHASE WHERE ent == 0 GROUP BY node_id")
    
    p = cursor.fetchall()
    m = len(p)
    n = len(p[0])
    q = np.zeros((m,n+1))
    startIndex = 0
    for i in range(m):
        q[i,0], q[i,1], q[i,3] = p[i][0], p[i][1], p[i][3]
        q[i,2] = min(100,p[i][2])/100
        q[i,4] = startIndex
        startIndex += q[i,3]
    conn.commit()
    conn.close()
    
    return (M, q)

def getQueuingModelParameters(database):
    """renvoie les paramètres du queuing model pour la simulation database
    database : string contenant le nom du fichier sqlite
    sortie : array [p, [k,gamma] ] taille N²,2*N"""
    
    conn = sqlite3.connect(database)
    
    cursor = conn.cursor()
    cursor.execute("select input_flow from ( select DISTINCT oid, lane, input_flow from MILANE where ent == 0 and sid = 0 )")
    gamma = np.array([g[0] for g in cursor.fetchall()])
    
    conn.commit()
    conn.close()
    
    return np.array([gamma, None, None])    