import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Percolation:

    def __init__(self, N):
        self.N = N
        self.openGrid = np.zeros(N ** 2, int)
        self.id = np.arange(0, N ** 2, 1)
        self.openSites = 0
        self.size = np.zeros(N ** 2, int)

    # Get 1D array index from row and column indexes
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

            if row != 0 and self.isOpen(row - 1, column):
                self.union(self.getLoc(row, column),
                           self.getLoc(row - 1, column))

            if row != (self.N - 1) and self.isOpen(row + 1, column):
                self.union(self.getLoc(row, column),
                           self.getLoc(row + 1, column))

            if column != 0 and self.isOpen(row, column - 1):
                self.union(self.getLoc(row, column),
                           self.getLoc(row, column - 1))

            if column != (self.N - 1) and self.isOpen(row, column + 1):
                self.union(self.getLoc(row, column),
                           self.getLoc(row, column + 1))

            return 0

    def numberOfOpenSites(self):
        return self.openSites

    # Check all the sites on the top row and see if they are connected
    # to the bottom row.
    def percolates(self):
        for j in np.arange(0, self.N, 1):
            for k in np.arange(self.N ** 2 - self.N, self.N ** 2, 1):
                if self.connected(j, k):
                    return True
        return False

    def plot(self):
        plt.figure()
        plt.imshow(self.returnGrid(), aspect='auto', cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.axes().set_aspect('equal')
        plt.show()

    def returnGrid(self):
        return np.reshape(self.openGrid, [self.N, self.N])

    def root(self, i):
        while i != self.id[i]:
            i = self.id[i]
        return i

    # Weighted quick-union
    def union(self, p, q):
        i = self.root(p)
        j = self.root(q)
        if i == j:
            return
        if self.size[i] < self.size[j]:
            self.id[i] = j
            self.size[j] += self.size[i]
        else:
            self.id[j] = i
            self.size[i] += self.size[j]

    def connected(self, p, q):
        return self.root(p) == self.root(q)


class PercolationStats:

    def __init__(self, N, trialCount):
        self.siteCount = np.zeros(trialCount)
        for i in xrange(trialCount):
            p = Percolation(N)
            while p.percolates() is False:
                row = np.random.randint(0, N)
                col = np.random.randint(0, N)
                p.open(row, col)
            self.siteCount[i] = p.numberOfOpenSites()

    def mean(self):
        return np.mean(self.siteCount)

    def stddev(self):
        return np.stddev(self.siteCount)


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


# Plot the percolation probability vs site occupation probability
# for a grid of size N, based on a number of random trials.
# This is estimated by a sigmoid curve fit on the data,
# but there are better ways of doing it...
def ProbabilityPlot(N, trialCount):
    pstats = PercolationStats(N, trialCount)

    counts = pstats.siteCount / N ** 2
    mean = pstats.mean() / N ** 2
    print 'Grid size: {}\t Mean: {}'.format(N, mean)
    # print counts

    plt.style.use('ggplot')
    values, base = np.histogram(counts, bins=40)
    cumulative = np.cumsum(values) / float(trialCount)

    xdata = base[:-1]
    ydata = cumulative

    popt, pcov = curve_fit(sigmoid, xdata, ydata)

    x = np.linspace(0, 1, 150)
    y = sigmoid(x, *popt)

    plt.xlim(0, 1)
    plt.plot(x, y, label='N = {}'.format(N))
    plt.ylim(0, 1.05)
    plt.axvline(mean, c='black', linestyle='dashed')


plt.figure()
ProbabilityPlot(10, 1000)
ProbabilityPlot(20, 100)
ProbabilityPlot(50, 10)
plt.legend(loc='best')
plt.savefig('outfile.png', dpi=270)
plt.show()

# N = 25
# p = Percolation(N)
# while p.percolates() is False:
#     row = np.random.randint(0, N)
#     col = np.random.randint(0, N)
#     p.open(row, col)

# p.plot()
