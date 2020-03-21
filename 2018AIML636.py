import sys
import os
import numpy as np

# Matrix
X = []
B = []

# Path of input
dir_path = os.path.dirname(os.path.realpath(__file__))
inputPath = dir_path + "/input.txt"

# Check if input filename is provided, else use input.txt from CWD
if len(sys.argv[1:]) > 0:
    inputPath = sys.argv[1]

# Open file, read line-by-line
loaded = np.loadtxt(inputPath)
rows, columns = loaded.shape[0], loaded.shape[1]

# Index matrix and get them into A, X and B
A = loaded[...,0:columns-1]
B = loaded[...,columns-1]

if A.shape[0] != A.shape[1]:
    print("Error : Input should lead to Square matrix, Found %s",str(A.shape))
    exit(0)

# Utility function
def det_2by2(nd):
    # calculate determinant for a 2x2 matrix
    if nd.shape[0] == nd.shape[1] == 2:
        #pass
        return ((nd[0,0] * nd[1,1]) - (nd[0,1] * nd[1,0]))
    else:
        print("Error : Only 2x2 array.")
    return None

# Utility function to get valid indexes for matric factorisation
def ValidIndex(row, col, shape):
    maxRow = shape[0]
    maxCol = shape[1]

    validRows = []
    validCols = []

    for x in range(0, maxRow):
        if x!=row:
            validRows.append(x)

    for y in range(0, maxCol):
        if y!=col:
            validCols.append(y)
    # Return indexes for factorisation
    return (np.array(validRows), np.array(validCols))

# Solving 'x' in Ax = B, x = Inv(A)B
# A matrix has inverse only if the determinant is non-zero
# Finding determinant of a Matrix

Ap = A[1:,1:]
Cp = A[1:,:2]
Bp = A[1:,[0,2]]

# a b c     =   d
# d e f     =   g
# h i j     =   k
# Determinat is defined as <D = a.A[1:,1:] - b.A[1:, [0,2]] + c.A[1:,2:]>
D = A[0,0]*det_2by2(Ap) - A[0,1]*det_2by2(Bp) + A[0,2]*det_2by2(Cp)

# Check if determinant is non-zero.
# If |D| of matrix is zero or less, the equations do not meet, and hence
# there exists no solution to the equation.
# To solve the equation, equation should result into lines that intersect 
# each other. And the determinant gives the volume/area of the graph intersection
# of the equations.

if D > 0:
    #Step1: Find adjacent matrix
    factors = np.ndarray((A.shape[0],A.shape[1]), dtype=float)
    for row in range(0,A.shape[0]):
        for col in range(0,A.shape[1]):
            # factors
            rows,cols = ValidIndex(row, col, A.shape)
            # debug
            _nd = A[rows,:][:,cols]
            factors[row,col] = det_2by2(_nd)
    #print(" Factors matrix : ", factors)
    
    factorsT = np.ndarray(factors.shape)
    #Step2: Find transpose of factor matrix
    for row in range(0,factors.shape[0]):
        for col in range(0,factors.shape[1]):
            factorsT[row][col] = factors[col,row]
    #print(" Factors Transpose : ", factorsT)
    #Step3: Dot product of sign matrix
    sign = np.array([[1,-1,1],[-1,1,-1],[1,-1,1]])
    factorsT = factorsT * sign
    #print(" Adjacent Matrix : ", factorsT)
    #Step4: Inv(A) = 1/|D| * Adj(A)
    factorsT = 1/D * factorsT
    #print(" Inverse Matrix : ", factorsT)
    #Step5: Finding solution
    # x = Inv(A)B
    print(B)
    x = np.dot(factorsT, B)
    #print("Answer : ", x)
    np.savetxt(dir_path + "/output.txt", x)
else:
    print("Matrix doesn't have Inverse")
    exit(0)