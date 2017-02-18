from random import randrange

class Deque:
    def __init__(self):
        self.items = []

    def is_empty(self):
        if len(self.items) == 0:
            return True
        else:
            return False

    def size(self):
        return len(self.items)

    def add_first(self, item):
        self.items.insert(0, item)

    def add_last(self, item):
        self.items.append(item)

    def remove_first(self):
        item = self.items[0]
        del self.items[0]
        return item

    def remove_last(self):
        item = self.items[-1]
        del self.items[-1]
        return item


class RandomisedQueue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        if len(self.items) == 0:
            return True
        else:
            return False

    def size(self):
        return len(self.items)

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        random_index = randrange(0, len(self.items))
        item = self.items[random_index]
        del self.items[random_index]
        return item

    def sample(self):
        random_index = randrange(0, len(self.items))
        return self.items[random_index]


print 'Deque testing'
print '---'
D = Deque()
D.add_last('Two')
D.add_last('One')
D.add_first('Zero')
print D.items
print D.size()
print D.remove_first()
print D.items
print D.remove_last()
print D.items
print

print 'Randomised queue testing'
print '---'
Q = RandomisedQueue()
Q.enqueue('One')
Q.enqueue('Two')
Q.enqueue('Three')
print Q.items
print Q.size()
print Q.sample(), Q.sample(), Q.sample()
print Q.items
print Q.dequeue()
print Q.items
print
