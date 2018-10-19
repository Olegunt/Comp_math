import numpy as np
import matplotlib.pyplot as plt

#=====================================================

def solve(L, U, P, b):
    
    #вводим замену y = Ux
    #решаем в два шага: 
    #1)Находим у из Ly = b 
    #2)Находим х из Ux = y   
    
    n = len(L)
    y = np.zeros(n)                     
    x = np.zeros(n)
    b = P.dot(b)
    
    L_inv = np.linalg.inv(L)          
    y = L_inv.dot(b)                    #решение уравнения Ly = b как y = L^-1*b                           
    U_inv = np.linalg.inv(U)
    x = U_inv.dot(y)                    #решение уравнения Ux = y как x = U^-1*y
    
    return x  

#=====================================================

def lu(A, permute):
    n = len(A)
    U = np.copy(A)
    L = np.eye(n)
    P = np.eye(n)
    
    for j in range(n - 1):
        P_i = np.eye(n)                 #i-ая матрица перестановок
        M = np.eye(n)                   #матрица множителей
        
        if permute:
            emax = np.copy(U[j][j])
            imax = j
            
            #поиск главного элемента в j столбце
            
            #for k in range (j, n):
            i = j
            while i < n:
                if U[i][j] > emax:
                    imax = i
                    emax = np.copy(U[i][j])
                i += 1
            
            #создание матрицы перестановок
            
            temp = np.copy(P_i[j])      #cохраняем текущую строку единичной матрицы
            
            P_i[j] = np.copy(P_i[imax]) #на место текущей строки записываем строку единичной матрицы, 
                                        #индекс которой равен индексу строки с главным элементом в матрице A
                                    
            P_i[imax] = temp            #переставляем текущую строку
            
            U = P_i.dot(U)              #делаем перестановку
            P = P_i.dot(P)              #матрица всех перестановок
        
        for i in range(j + 1, n): 
            m = U[i][j] / U[j][j]
            M[i][j] = -m
            
        U = M.dot(U)                    #итерация метода Гаусса
        if permute:
            L = (M.dot(P_i)).dot(L)
        else:
            L = M.dot(L)
            
    L = P.dot(np.linalg.inv(L))    

    return L, U, P

#=====================================================   
        
A = np.array([[3., 1.,-3.], [6., 2., 5.], [1., 4., -3.]])
b = np.array([-16., 12., -39.])
p = np.linspace(0,12,13) 
E_1 = np.zeros(len(p))
E_0 = np.zeros(len(p))

L, U, P = lu(A,1)
x = solve(L, U, P, b)  # точное решение
print(x)

for i in range(len(p)):
    A_mod = np.copy(A)
    A_mod[0][0] += 10**-(p[i])
    
    #использование перестановок (частичный выбор главного элемента)
    
    L_1, U_1, P_1 = lu(A_mod,1) 
    b_1 = np.array([-16., 12., -39.])
    b_1[0] += 10**-(p[i])
    x_1 = solve(L_1, U_1, P_1, b_1)
    E_1[i] = max(np.abs(x - x_1)) / max(np.abs(x))
    
    #без использования перестановок (без частичного выбора главного элемента)
    
    L_0, U_0, P_0 = lu(A_mod,0)
    b_0 = np.array([-16., 12., -39.])
    b_0[0] += 10**-(p[i])
    x_0 = solve(L_0, U_0, P_0, b_0)
    E_0[i] = max(np.abs(x - x_0)) / max(np.abs(x))
    
#=====================================================    
    
fig, axes = plt.subplots(figsize = (6, 6))    
axes.semilogy(p, E_1, 'ob', label= "$E_{permute}$")
axes.semilogy(p, E_0, 'or', label="$E$")
axes.legend(loc='upper left')
axes.set_xlabel('${p}$')
axes.set_ylabel('$E$')
axes.grid(True)

fig.savefig("lab3.pdf")
