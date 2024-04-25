import numpy as np

# matrix definitions
A = np.array([[0, 1, 0, 0],[0,0,-1,0],[0,0,0,1],[0,0,5,0]])
b = np.array([0,1,0,-2])
c = [1,0,0,0]

# print the A matrix
print("A")
print(A)
print("A^2")
print(A@A)
print("A^3")
print(A@A@A)

# compute the controlability matrix
C = np.vstack((b, A@b, A@A@b, (A^3)@b))
print("C")
print(C)

rank = np.linalg.matrix_rank(C)
print("Rank of C: {0}".format(rank))
if C.shape[0] == rank:
    print("This is full rank, the system is controllable! :D")
else:
    print("The system is not controllable")