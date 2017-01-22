import numpy as np

class Percolation:

    def __init__(self, N):
        self.N = N
        self.openGrid = np.zeros(N ** 2, int)
        self.id = np.arange(0, N ** 2, 1)
        self.openSites = 0

    def getLoc(self, row, column):
        return row * self.N + column

    def isOpen(self, row, column):
        return self.openGrid[self.getLoc(row, column)] == 1

    def isFull(self, row, column):
        return self.openGrid[self.getLoc(row, column)] == 0

    def open(self, row, column):
        if self.isOpen(row, column):
            return 1
        else:
            self.openGrid[row * self.N + column] = 1
            self.openSites += 1

            if row != 0 and self.isOpen(row - 1,  column):
                self.union(self.getLoc(row, column), self.getLoc(row - 1, column))

            if row != (self.N - 1) and self.isOpen(row + 1,  column):
                self.union(self.getLoc(row, column), self.getLoc(row + 1, column))

            if column != 0 and self.isOpen(row,  column - 1):
                self.union(self.getLoc(row, column), self.getLoc(row, column - 1))

            if column != (self.N - 1) and self.isOpen(row,  column + 1):
                self.union(self.getLoc(row, column), self.getLoc(row, column + 1))

            return 0

    def numberOfOpenSites(self):
        return self.openSites

    def percolates(self):
        for j in np.arange(0, self.N, 1):
            for k in np.arange(self.N ** 2 - self.N, self.N ** 2, 1):
                if self.connected(j, k):
                    return True
        return False

    def returnGrid(self):
        return np.reshape(self.openGrid, [self.N, self.N])

    def root(self, i):
        while i != self.id[i]:
            i = self.id[i]
        return i

    def union(self, p, q):
        i = self.root(p)
        j = self.root(q)
        self.id[i] = j

    def connected(self, p, q):
        return self.root(p) == self.root(q)

N = 10
p = Percolation(N)

while p.percolates() == False:
    row = np.random.randint(0, N)
    col = np.random.randint(0, N)
    p.open(row, col)

print p.id
print p.percolates()
print p.returnGrid()
