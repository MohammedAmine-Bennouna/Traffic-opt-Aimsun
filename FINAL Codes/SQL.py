# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:54:27 2017

@author: yassine
"""

import sqlite3 as sql

file = 'network'

#Ouvrir la base de données si elle existe sinon la créer.
conn = sql.connect(file)

#Définir un cursor pour passer envoyer les requettes.
cursor = conn.cursor()

#cursor.execute('CREATE TABLE users (id INT, name TEXT, age INT, primary key(id));')

cursor.execute("select input_flow from ( select DISTINCT oid, lane, input_flow from MILANE where ent == 0 and sid = 0 )")

print(cursor.fetchall())

conn.commit()

conn.close()