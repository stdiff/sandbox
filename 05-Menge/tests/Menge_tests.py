from nose.tools import *
from Menge.Menge import Menge
import random


def random_list_1(k=30):
    return([random.randint(0,20) for _ in range(k)])

def random_list_2(k=None):
    if k is None:
        k = random.randint(5,15)
    return([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(k)])


def test_constructor():
    random.seed(3)

    for list_gen in [random_list_1,random_list_2]:
        for _ in range(100):
            lst = list_gen()
            A = Menge(lst)
            S = set(lst)
            
            assert_equal(sorted(list(A)), sorted(list(S)))
            assert_equal(len(A), len(S))

            B = A.copy()
            assert_not_equal(id(A), id(B)) ## addresses must be different
            assert_equal(sorted(list(A)), sorted(list(B)))

            for a in 'abcdefghijklmnopqrstuvwxyz':
                assert_equal(a in lst, a in A)

            for x in range(20):
                assert_equal(x in lst, x in A)


def test_binary_relations():
    random.seed(19)
    
    for list_gen in [random_list_1,random_list_2]:
        for _ in range(100):
            small_lst = list_gen(10)
            large_lst = list_gen(30)

            A = Menge(small_lst)
            B = Menge(large_lst)

            S = set(small_lst)
            T = set(large_lst)

            assert_equal(A.issubset(B), S.issubset(T))
            assert_equal(B.issuperset(A), T.issuperset(S))


def test_unary_operations():
    random.seed(2)

    for list_gen in [random_list_1,random_list_2]:
        for _ in range(100):
            samples = set([2,3,5,7,11,13,17,19,'a,k,s,t,n,h,m,y,r,w'])

            lst = list_gen()
            A = Menge(lst)
            S = set(lst)

            x = samples.pop()
            A.add(x)
            S.add(x)

            assert_equal(sorted([str(x) for x in list(A)]),
                         sorted([str(x) for x in list(S)]))

            x = samples.pop()
            try:
                A.remove(x)
                eA = 'OK'
            except KeyError:
                eA = 'BAD'

            try:
                S.remove(x)
                eS = 'OK'
            except KeyError:
                eS = 'BAD'

            assert_equal(eA,eS)

            x = samples.pop()
            A.discard(x)
            S.discard(x)

            x = A.pop()
            assert x in S ## if an element in A popped out
            assert_equal(len(A)+1,len(S)) ## if an element is removed

            A.clear()
            assert_equal(len(A),0)


def test_binary_operations():
    random.seed(6)

    for list_gen in [random_list_1,random_list_2]:
        for _ in range(100):    
            lst1 = list_gen()
            lst2 = list_gen()

            A = Menge(lst1)
            B = Menge(lst2)

            S = Menge(lst1)
            T = Menge(lst2)

            assert_equal(sorted(list(A.intersection(B))),
                         sorted(list(S.intersection(T))))

            assert_equal(sorted(list(A.union(B))),
                         sorted(list(S.union(T))))

            assert_equal(sorted(list(A.difference(B))),
                         sorted(list(S.difference(T))))
            
            assert_equal(sorted(list(B - A)), sorted(list(T-S)))
