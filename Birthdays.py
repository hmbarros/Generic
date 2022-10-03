#Probabilidade de 2 terem a mesma idade em uma sala

n = int(input("Digite o numero de pessoas do grupo: "))
a = 1
for i in range(1,n):
  a = a*((366-i))/365
b = round((1-a)*100, 2)
print(f"A probabilidade é de {b}%")

a = 1
i = 1
n = int(input("Escolha o em qual casa deseja arredondar a porcentagem: "))

while True:
  a = a*((366-i))/365
  b = round((1-a)*100, n)
  print(f"Para {i} pessoas a probabilidade é de {b}%")
  i = i + 1
  if b == 100:
    break