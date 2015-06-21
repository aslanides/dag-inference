import numpy as np
from message_passing import *


p_b = np.array([0.97, 0.01, 0.02])
p_a_given_b = np.array([[0.9,0.8,0.3],[0.1,0.2,0.7]])

f_ab = p_a_given_b * p_b
f_bcd = np.array([[[0.9,0.08,0.01,0.01],[0.8,0.17,0.01,0.02],[0.1,0.01,0.87,0.02]],
                  [[0.3,0.05,0.05,0.6],[0.4,0.05,0.15,0.4],[0.01,0.01,0.97,0.01]]])
f_c = np.array([0.7,0.3])
f_de = np.array([[0.99,0.99,0.4,0.9],[0.01,0.01,0.6,0.1]])
f_cf = np.array([[0.99,0.2],[0.01,0.8]])

A = Variable(['F','T'],None)
f_AB = Factor(f_ab,[A],{A : 0})
B = Variable(['none','mild','severe'],[f_AB])
F = Variable(['F','T'],None)
f_CF = Factor(f_cf,[F],{F : 0})
f_C = Factor(f_c,None,None)
C = Variable(['F','T'],[f_CF,f_C])
f_BCD = Factor(f_bcd,[B,C],{B : 1,C : 0})
D = Variable(['healthy', 'carrier', 'sick', 'recovering'],[f_BCD])
f_DE = Factor(f_de,[D],{D : 1})
E = Variable(['F','T'],[f_DE])

print factor_to_variable_message(f_DE,E)
