# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:42:25 2017

@author: Yassine
"""


def findNode( model, entry ):
    node = model.getCatalog().find( int(entry) )
    if node != None:
        if node.isA( "GKNode" ) == False:
            node = None
    return node

def findControlPlan( model, entry ):
    controlPlan = model.getCatalog().find( int(entry) )
    if controlPlan != None:
        if controlPlan.isA( "GKControlPlan" ) == False:
            controlPlan = None

    return controlPlan



def phaseChange(model, id_n, id_CP, phases):
    '''modifie les durées des phases du noeud id_n, selon la liste phases [from, duration]'''

        
    node = findNode(model, id_n)
    controlPlan = findControlPlan(model,id_CP)
    controlJunction = controlPlan.createControlJunction(node)
    listControlPhase = controlJunction.getPhases()
    l = len(listControlPhase)
    
    #print("Modifying Node {} CP {} \n".format(int(id_n),int(id_CP)))   
        
    
        
        
        
    for i in range(l):
        controlPhase = listControlPhase[i]
        
        fromTime = phases[i][0]
        duration = phases[i][1]
      
        controlPhase.setFrom(fromTime)
        controlPhase.setDuration(duration)

    