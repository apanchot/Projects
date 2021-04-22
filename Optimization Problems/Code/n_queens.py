import numpy as np

class NQueens():

    def __init__(self, data, n_pop=20, n_iter=100, return_unique=True):
        self.data = data  # data matrix
        self.n_iter = n_iter  # number of runs
        self.n_pop = n_pop  # number of solutions returned
        self.ru = return_unique  # return unique best solutions (if possible); otherwise return best even if they repeat

    def search(self):
        try:
            N = len(self.data[0])
        except:
            raise ValueError("There must be at least one city")
        if N > 1:

            beggin = 0
            while 2 ** beggin < self.n_iter and beggin < N - 2:
                beggin += 1
            pops = []
            summss = []
            for k in range(self.n_iter):
                ct = 0
                sol = np.zeros(N, dtype=int)
                np.random.seed(k)
                board = np.identity(N, dtype=int)
                row = 0
                board[:, 0] = 1
                for j in range(0, beggin):
                    avail = np.reshape(np.argwhere(board[row] == 0), -1)
                    low = np.argsort(data[row, avail])
                    try:
                        da = np.random.choice(low[:2])
                    except:
                        da = low[0]
                    a = avail[da]
                    board[:, a] = 1
                    board[row, a] = 8
                    sol[ct] = row
                    row = a
                    ct += 1
                for j in range(beggin, N - 1):
                    avail = np.reshape(np.argwhere(board[row] == 0), -1)
                    low = np.argsort(data[row, avail])
                    da = low[0]
                    a = avail[da]
                    board[:, a] = 1
                    board[row, a] = 8
                    sol[ct] = row
                    row = a
                    ct += 1
                board[row, 0] = 8
                sums = 0
                sol[ct] = row
                for i in range(N):
                    for j in range(N):
                        if board[i, j] == 8:
                            sums += data[i, j]
                summss.append(sums)
                pops.append(sol)
            summssin = np.argsort(summss)  # sort by fittest solutions
            popsin = []

            if self.ru:  # find unique solutions
                if len(np.unique(summss, axis=0)) >= self.n_pop:
                    i = 0
                    while len(popsin) < self.n_pop and i < self.n_iter:
                        if list(pops[summssin[i]]) not in popsin:
                            popsin.append(list(pops[summssin[i]]))
                        i += 1
                    if (len(popsin) == self.n_pop):
                        return (popsin)
                    i = 0
                    while len(popsin) < self.n_pop:
                        popsin.append(list(pops[summssin[i]]))
                        i += 1

                    return (popsin)
                else:
                    for i in range(self.n_pop):
                        popsin.append(list(pops[summssin[i]]))
                    return (popsin)
            else:  # ignore unique requirement
                for i in range(self.n_pop):
                    popsin.append(list(pops[summssin[i]]))
                return (popsin)
        else:  # if N ==1
            a = []
            for i in range(self.n_pop):
                a.append([0])
            return a