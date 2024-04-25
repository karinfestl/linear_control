import numpy as np

def controlability_matrix(A, b): # TODO: we can easily generalize this to nxn matrix
    if len(b) == 4:
        C = np.hstack((b, A @ b, A @ A @ b, (A @ A @ A) @ b))
    if len(b) == 3:
        C = np.hstack((b, A @ b, A @ A @ b))
    if len(b) == 2:
        C = np.hstack((b, A @ b))
    return C


def characteristic_polynomial(A):
    eigenvalues = np.linalg.eigvals(A)
    alpha = np.poly(eigenvalues)  # Find the coefficients of a polynomial with the given sequence of roots.
    print("lambda = {0}".format(np.polynomial.Polynomial.fromroots(eigenvalues)))
    return alpha


if __name__ == '__main__':
    ''' problem statement '''
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 5, 0]])
    b = np.array([[0, 1, 0, -2]]).T
    c = [1, 0, 0, 0]
    eigenvalues_f = [complex(-1.5, 0.5), complex(-1.5, -0.5), complex(-1, 1), complex(-1, -1)]  # desired eigenvalues

    ''' compute transformation '''
    C = controlability_matrix(A, b)
    alpha = characteristic_polynomial(A)

    Cbarinv = np.eye(4)
    Cbarinv[0, 1] = Cbarinv[1, 2] = Cbarinv[2, 3] = alpha[1]
    Cbarinv[0, 2] = Cbarinv[1, 3] = alpha[2]
    Cbarinv[0, 3] = alpha[3]

    Pinv = C @ Cbarinv
    P = np.linalg.inv(Pinv)

    ''' compute feedback gain '''
    alphabar = np.poly(eigenvalues_f)

    kbar = alphabar[1:] - alpha[1:]
    kbar = np.expand_dims(kbar, 0)
    k = kbar @ P

    ''' output results '''
    print("C")
    print(C)
    print("Cbarinv")
    print(Cbarinv)
    print("Pinv")
    print(Pinv)
    print("P")
    print(P)
    print("kbar = {0}".format(kbar))
    print("k = {0}".format(k))

    Afbar = P @ (A - b @ k) @ Pinv  # controlled dynamics in control canonical form
    print("Afbar")
    print(np.round(Afbar, decimals=2))
