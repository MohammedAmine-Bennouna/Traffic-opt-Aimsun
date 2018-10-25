# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import numpy.linalg as LA

import random as rd

import scipy as sp
import scipy.optimize as sco

from sympy import symbols, nsolve

import Data_Extractor as data

import os

#==============================================================================
# Paramètres du problème
#==============================================================================

path = "D:\2A\PSC\Code_Python"
file = 'network.sqlite'
file2 = 'complete.sqlite'

def Aimsun_N():
    '''renvoie le nombre de files dans le métamodèle'''
    
    N = data.getLanesNumber(file2)
    return N

def Aimsun_M():
    '''renvoie le nombre de noeuds du réseau'''
    
    M = data.getNetworkParameters(file2)[0]
    return M

def Aimsun_p():
    '''renvoie le paramètre exogène p d'Aimsun
    sortie : [id_CP, id_n, cyc. dur., avail. cyc. ratio, nb of phases, start index]'''
    
    p = data.getNetworkParameters(file2)[1]
    return p

def Aimsun_q():
    '''renvoie le paramètre exogène q du queuing model
    sortie : array (p,(k,gamma)) taille N²,2*N'''
    
    q = data.getQueuingModelParameters(file2)
    return q


p = Aimsun_p()
q = Aimsun_q()
N = Aimsun_N()
M = Aimsun_M()

w0 = 1 # poids de la composante physique dans la régression quadratique
rMin = 0.05 # ratio minimal d'une phase d'un cycle sémaphorique
totDur = 90 # durée d'un cycle de feux

#==============================================================================
# Fonction objectif et modèles
#==============================================================================

def Aimsun_simul(x,p,aim):
    '''x : variable de contrôle array [phases ratio]
    p : paramètre du réseau
    aim : objet de classe Connexion ; envoie les requêtes à Aimsun
    lance une simulation sur Aimsun, pour le plan de feux x'''
    
    params = []
    
    for i in range(M):
        
        sI = int(p[i,4])
        nbP = int(p[i,3])
        phases = x[sI:sI+nbP]
        id_node = p[i,1]
        id_CP = p[i,0]
        param = [ id_node, id_CP, [] ]
        
        start = 0
        for k in range(nbP):
            duration = phases[k]*totDur
            param[2].append([start, duration])
            start += duration
        params.append(param)
        
    aim.setParams(params)
    aim.simulate()
    
    return None

def f(x,p,aim):
    '''x : variable de contrôle
    p : paramètre du réseau
    renvoie le temps de trajet total pour (x,p) calculé avec Aimsun'''
    
    Aimsun_simul(x,p,aim)

    TTT = data.getTotalTravelTime(file)
    return TTT


def calc_mu(x):
    '''x : variable de contrôle array taille M'''
    return None

def queuing_model(x,q):
    '''x : variable de contrôle array taille M
    renvoie le y associé à x par le queuing model'''
    mu = calc_mu(x)
    
    var = []
    eqn = []
    
    for i in range(N):
        
        lambda_eff = symbols('lambda{}'.format(i))
        rho_eff = symbols('rho{}'.format(i))
        P = symbols('P{}'.format(i))
        
        var.append(lambda_eff)
        var.append(rho_eff)
        var.append(P)
       
    for i in range(N):
        
        lambda_eff = var[3*i]
        rho_eff = var[3*i+1]
        P = var[3*i+2]
        
        gamma = q[1][i][1]
        k = q[1][i][0]
        
        eqn1 = -lambda_eff + gamma * (1-P)
        
        for j in range(N):
            eqn1 += q[0][j][i] * var[3*j]
        
        eqn2 = - rho_eff + lambda_eff / mu[i]
        S1 = 0
        S2 = 0
        for j in range(N):
            p = q[0][i][j]
            S1 += p * var[3*j+2]
            if (p != 0):
                S2 += var[3*j+1]
        eqn2 += S1 * S2
        
        eqn3 = -P + (1-rho_eff)/(1-rho_eff**(k+1)) * rho_eff**k
        
        eqn.append(eqn1)
        eqn.append(eqn2)
        eqn.append(eqn3)
    
    y0 = np.array([0] * (3*N))
    sol = nsolve(eqn,var,y0)
    
    y = np.zeros((N,3))
    for i in range(N):
        y[i][0] = sol[3*i]
        y[i][1] = sol[3*i+1]
        y[i][2] = sol[3*i+2]
    
    return y
        

def T(x,y,q):
    '''x : variable de contrôle
    y : variable du modèle de file d'attente array (lambda,rho,P) taille 3*N
    q : paramètre du modèle de file d'attente array (p,(k,gamma)) taille N²,2*N
    renvoie le temps de trajet total pour (x,y,q) du modèle analytique'''
    numer = 0
    denom = 0
    for i in range(N):
        
        rho = y[i][1]
        k = q[1][i][0]
        numer += rho / (1-rho) - (k+1)*rho**(k+1) / (1-rho**(k+1))
        
        gamma = q[1][i][1]
        P = y[i][2]
        denom += gamma * (1-P)
    
    return numer / denom

def Phi(x,beta):
    '''x : variable de contrôle
    beta : coefficients du polynôme
    calcule la composante fonctionnelle du métamodèle'''
    l = len(x)
    res = beta[0]
    for i in range(l):
        res += (beta[l+i+1] * x[i] + beta[i+1]) * x[i]
    return res

def m(x,y,alpha,beta,q):
    '''x : variable de contrôle
    y : variable du modèle de file d'attente
    q : paramètre du modèle de fil d'attente
    (alpha, beta) : paramètres du métamodèle'''
    if (y == None):
        return Phi(x,beta)
    else:    
        return alpha * T(x,y,q) + Phi(x,beta)

#==============================================================================
# Criticality step
#==============================================================================

def stat(x,y):
    '''renvoie la mesure de stationnarité évaluée en (x,y)'''
    return None

def cons_mode(x,Delta):
    '''renvoie un modèle certifié fully linear en x'''
    return None

#==============================================================================
# Step calculation
#==============================================================================

def somI(i,x):  
    ''' contrainte (10) de l'article d'Osorio'''
    som=0
    sI = int(p[i][4])
    nbP = int(p[i][3])
    xI = x[sI : sI+nbP]
    for j in range(nbP):
        som = som + xI[j]
    return som
    
    
def h2(x,y,q):
    ''' pour la contrainte (11) dans l'article d'Osorio'''
    l=np.array([])
    
    for i in range(N):
        gamma = q[1][i][1]
        k = q[1][i][0]
        
        S1=0
        S2=0
        S3=0
       
        for j in range(N):
            S1 += q[0][j][i] * y[3*j]
            S2 += q[0][i][j] * y[3*j+1]   
            S3 += y[3*j+2]  
        
        h1 = -y[3*i]-gamma*(1-y[3*i+2])+ S1
        h2 = -y[3*i+1]-y[3*i]/calc_mu(x)[i]+S2*S3
        h3 = y[3*1+2] - (1-y[3*i+1])*(y[3*i+1])**(k)/(1-y[3*i+1]**(k+1))
        
        l = np.append(l,h1)
        l = np.append(l,h2)
        l = np.append(l,h3)
        
    return l


def min_TR(x,y,delta,alpha,beta,q,qMod):
    
    '''(x, Delta) : région de confiance
    (alpha, beta) : paramètres du métamodèle
    minimise le métamodèle sur la région de confiance'''
    
    l1 = len(x)
    
    if (y != None):
        l2 = len(y)
        X = np.concatenate((x,y))
    else:
        X = x
    
    if qMod:
        
        def m_temp(X):
            return m(X[0:l1],X[l1:l1+l2],alpha,beta,q)
    
    else:
        
        def m_temp(X):
            return m(X[0:l1],None,alpha,beta,q)
        
    cons = [] # liste des contraintes
    
    # Contrainte 1: durée du cycle - Osorio (10)
    for i in range(M):
        sI = int(p[i][4])
        nbP = int(p[i][3])
        minRat = p[i][2]
        l = {'type': 'ineq',
           'fun': lambda X: -abs(sum(X[sI:sI+nbP])-minRat)}
        cons.append(l)
    
    # Contrainte 2: région de confiance - Osorio (12)
    l={'type': 'ineq',
           'fun': lambda X: delta-np.linalg.norm(X[0:l1]-x)}
    cons.append(l)
    
    # Contraintes 3 & 4: minimum pour x et y - Osorio (13-14)
    if qMod:
        for i in range(l2):
            l={'type': 'ineq',
               'fun': lambda X: X[l1:l1+l2]}
            cons.append(l)
    
    for i in range(l1):
        b={'type': 'ineq', 'fun': lambda X: X[0:l1]- np.array([rMin for k in range(l1)])}   
        cons.append(b)
        
    # Contrainte 5 : y suit le queuing model - Osorio (11)
    if qMod:
        for k in range(N):
            l1={'type': 'ineq',
               'fun': lambda X: -abs( h2(X[0:l1],X[l1:l1+l2],q)[3*k] )}
            l2={'type': 'ineq',
               'fun': lambda X: -abs( h2(X[0:l1],X[l1:l1+l2],q)[3*k+1] )}
            l3={'type': 'ineq',
               'fun': lambda X: -abs( h2(X[0:l1],X[l1:l1+l2],q)[3*k+2] )}
            cons.append(l1)
            cons.append(l2)
            cons.append(l3)

    res = sco.minimize(m_temp, X, method = 'COBYLA', constraints = cons)
                        
    # si on pose res=min_TR(xk,y,beta,delta,alpha,q),
    # res.x est le point en lequel le minimum est atteint
    
    if qMod:
        return (res.x[0:l1],res.x[l1+l2])
    else:
        return (res.x[0:l1],None)

#==============================================================================
# Acceptance of the trial point
#==============================================================================

#==============================================================================
# Model improvement
#==============================================================================

def random(p):
    '''renvoie une variable de contrôle tirée au hasard et compatible avec les
    paramètres p du réseau'''
    res = []
    for i in range(M):
        nbP = int(p[i][3])
        avCycR = p[i][2]
        resI = [rd.random() for k in range(nbP)]
        s = sum(resI)
        resI = [(avCycR - nbP*rMin)/s * x + rMin for x in resI]
        res += resI
    return np.array(res)

def ecart_rel(alpha,beta,alpha_new,beta_new):
    '''renvoie l'écart relatif entre les paramètres (alpha,beta) old/new'''
    old = np.concatenate((alpha,beta))
    new = np.concatenate((alpha_new,beta_new))
    return LA.norm(old-new) / LA.norm(old)

def weights(x,sample):
    '''x : itérée en cours
    sample : ensemble des points où f(x,p) est connue
    met à jour les poids utilisés pour la régression quadratique de fit'''
    length = len(sample)
    weights_array = [ 1/(1 + LA.norm(x - sample[i][0])) for i in range(length) ]
    return weights_array


def fun_to_minimize(alpha_beta_couple, current, sample, q, qMod):
    '''fonction à minimiser pour la régression quadratique (fit)'''
    result = 0
    alpha, beta = alpha_beta_couple[0], tuple(alpha_beta_couple[1:])
    length = len(sample)
    d = len(beta)
    weights_array = weights(current[0], sample)
 
    for i in range(length):
        x_i = sample[i][0]
        y_i = sample[i][1]
        result += (weights_array[i]*(sample[i][2]-m(x_i,y_i,alpha,beta,q)))**2
        
    for i in range(d):
            result += (w0*beta[i])**2
    
    if qMod:    
        result += (w0*(alpha-1))**2
        
    return result 


def fit(sample, current, q, qMod):
    '''sample : ensemble des évaluations (x,y,f(x)) disponibles
    renvoie le couple (alpha,beta) permettant de fit le métamodèle par
    régression quadratique'''
    l = len(current[0])
    x0 = np.array( [1]+[0]*(2*l+1) )
    print("Fitting...")
    alpha_beta_couple_opt = sp.optimize.least_squares(fun_to_minimize, x0,  args=(current, sample, q, qMod)).x
    print("Fitting finished.")
    alpha,beta = alpha_beta_couple_opt[0:1], alpha_beta_couple_opt[1:]
    return (alpha,beta)

#==============================================================================
# Algorithme d'optimisation
#==============================================================================

def optimise(x0, p, q, nInit, nMax, aim, Delta0=10, r=1, epsC=10**(-6),
             eta=10**(-3), tau_inf=0.1, gamma_inc=1.2, Delta_max=10**3,
             u_max=3, gamma=0.9, Delta_min=10**(-2), cMode = False, qMod = False):
    '''minimise la fonction objectif f dans la limite de nMax simulations en
    partant de la région de confiance (x0,Delta0)
    
    x0 : variable initiale
    (p,q) : paramètres du réseau
    Delta0 : rayon de confiance initial
    nInit : nombre de points pour initialiser le métamodèle
    nMax : nombre maximal de simulations 
    aim : objet de classe Connexion ; envoie les requêtes à Aimsun
    r : nombre de simulations pour estimer f
    epsC : stationnarité minimale
    eta : ratio minimal variation de f / variation de m
    tau_inf : variation minimale de (alpha,beta)
    gamma_inc : taux d'augmentation du rayon de confiance
    Delta_max : rayon de confiance maximal
    u_max : nombre maximal de rejet de nouvelles itérées
    gamma : taux de diminution du rayon de confiance
    Delta_min : rayon de confiance minimal
    cMode : utilisation du mode conservatif
    qMod : utilisation du modèle de files d'attente'''
    
#   Initialisation
    Delta = Delta0
    
    listSet = []
    
    if qMod:
        y0 = queuing_model(x0,q)
        TTT = f(x0,p,aim)
        print(TTT,"\n")
        sample = [[x0,y0,TTT]] # On initialise avec nInit points -> premier métamodèle fonctionnel
        current = sample[0]
        
        (alpha,beta) = (1,[0]*(2*M+1))
        
    else:
        TTT = f(x0,p,aim)
        sample = [[x0,None,TTT]]
        print(TTT,"\n") # On initialise avec nInit points -> premier métamodèle fonctionnel
        
        for k in range(nInit):
            x = random(p)
            y = None
            TTT = f(x,p,aim)
            listSet.append(TTT)
            print(TTT,"\n")
            sample.append([x,y,TTT])
        
        tmp = [l[2] for l in sample]
        k = tmp.index(min(tmp))
        current = sample[k]
        x = current[0]
        
        n = nInit*r # nombre de simulations effectuées
        (alpha,beta) = fit(sample,current,q,False) # initialisation du métamodèle

    u = 0 # nombre de trial points refusés
    
    cpt = 0 # compteur de boucles
    
    listRes = [] #mémoire des TTT
    
    while(n<nMax):
        
        cpt += 1
        print("Loop {}. \n".format(cpt))
    
#   Step 1 - Criticality step
        if cMode and (stat(x,y) < epsC):
            (alpha,beta,Delta) = cons_mode(x,Delta)
    
#   Step 2 -  Step calculation
        print("Calculating step... \n")
        (x_new,y_new) = min_TR(x,y,Delta,alpha,beta,q,qMod) 

#   Step 3 Acceptance of the trial point
        f0 = current[2]
        f1 = f(x_new,p,aim)
        n += r
        m0 = m(x,y,alpha,beta,q)
        m1 = m(x_new,y_new,alpha,beta,q)
        
        rho = (f0 - f1) / (m0 - m1)
        
        if (rho > eta):
            x = x_new
            y = y_new
            current = [x,y,f1]
            u = 0
            print("Trial point accepted. \n")
        else:
            u += 1
            print("Trial point refused. \n")
    
        sample.append([x_new,y_new,f1])

#   Step 4 - Model improvement
        if (not qMod) or (qMod and (n > nInit)):
            
            (alpha_new, beta_new) = fit(sample,current,q,qMod) # mise à jour du métamodèle
            
            tau = ecart_rel(alpha,beta,alpha_new,beta_new)
            
            if(tau < tau_inf):
                print("Additional fit required. Improving metamodel... \n")
                x_new = random(p)
                if qMod:
                    y_new = queuing_model(x_new,q)
                else:
                    y_new = None
                f1 = f(x_new,p,aim)
                n += r
                sample.append([x_new,y_new,f1])
                (alpha,beta) = fit(sample,current,q,qMod)
                print("Metamodel improved. \n")
            
            else:
                (alpha,beta) = (alpha_new,beta_new)
                print("No metamodel improvement required. \n")

# Step 5 - Trust region radius update
        if(rho > eta):
            Delta = min(gamma_inc * Delta, Delta_max)
            print("Trust radius increased. \n")
            
        elif(u > u_max):
            Delta = max(gamma * Delta, Delta_min)
            print("Trust radius decreased. \n")
            u = 0
        
        if cMode and (Delta <= Delta_min):
            (alpha,beta,Delta) = cons_mode(x,Delta)
        
        print("Total travel time at this point: {} \n".format(current[2]))
        listRes.append(current[2])
            
    return listSet, listRes

if __name__ == "__main__":
    print('')