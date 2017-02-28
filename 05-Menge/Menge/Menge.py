#!/usr/bin/python3 -tt

class Menge():
    def __init__(self,elm=None):
        self.menge = {}

        ## the following attributes are used for a generator.
        self.lst = []
        self.index = 0

        if elm is not None:
            for x in elm:
                self.menge[x] = 1

    def __iter__(self):
        self.index = 0
        self.lst = list(self.menge.keys())
        return self

    def __next__(self):
        try:
            result = self.lst[self.index]
            self.index += 1
            return result
        except IndexError:
            raise StopIteration

    def __repr__(self):
        '''
        This provides an "official" string for an instance 
        '''
        lst = []
        for x in list(self):
            s = "'%s'" % x if isinstance(x,str) else str(x)
            lst.append(s)
        string = ', '.join(lst)
        return("{%s}" % string)

    def __len__(self):
        '''
        We can apply len() function to an instance
        '''
        return(len(self.menge.keys()))

    def __nonzero__(self):
        '''
        we can use an instance of this class as a condition 
        '''
        return(len(self.menge.keys()) > 0)


    def copy(self):
        return Menge(self)

    def issubset(self,B):
        for a in self:
            if a not in B:
                return False
        return True

    def issuperset(self,B):
        return B.issubset(self)

    def add(self,x):
        self.menge[x] = 1
        return self

    def remove(self,x):
        if x in self.menge.keys():
            del self.menge[x]
        else:
            raise KeyError(x)
        return self

    def discard(self,x):
        if x in self.menge.keys():
            del self.menge[x]
        return self

    def pop(self):
        x = list(self.menge.keys())[0]
        del self.menge[x]
        return x

    def clear(self):
        self.menge = {}
        return self

    def intersection(self,B):
        C = Menge()
        for a in self:
            if a in B:
                C.add(a)
        return C

    def union(self,B):
        C = self.copy()
        for b in B:
            C.add(b)
        return C

    def difference(self,B):
        C = self.copy()
        for b in B:
            C.discard(b)
        return C

    def __sub__(self,B):
        return self.difference(B)
