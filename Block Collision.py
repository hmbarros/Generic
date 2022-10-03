import math
from copy import deepcopy

n = int(input("Quantas centenas de vezes o bloco 2 será maior que o bloco 1: "))
vi = [0, -10000]
m =  [1, 100**n]
vf = [0, -10000]
x = 0
while vf[0]>vf[1] or vf[0]<0:
  #print(vi[0]>vi[1] or vi[0]<0)
  #print(vf)
  vf[0] = (abs(vi[0])*(m[0]-m[1])+2*vi[1]*m[1])/(m[0]+m[1])
  vf[1] = (vi[1]*(m[1]-m[0])+2*abs(vi[0])*m[0])/(m[1]+m[0])
  vi = deepcopy(vf)
  if vi[0] < 0:
    vi[0] = -vi[0] 
    x += 1
    #print("!")
  x += 1
  #print("?")
print("O numero de colisões será:", x)
print(x)