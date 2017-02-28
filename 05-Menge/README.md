# Menge

Just a practice of writing a class module and a test.

This module provides a subset of functionalities of the built-in type `set`.
(cf. [Set Types](https://docs.python.org/3.4/library/stdtypes.html#set-types-set-frozenset))


## Requirements 

- Each element of a set must be hashable.
- A constructor takes an iterable object and create an instance. 
- An instance is a generator. But the order of emitted elements is not fixed.
- An instance can give a string representation like `set`.
- `len(A)` returns the cardinal number of the set.

### methods

- `copy()` : returns a new instance which has the same elements.

The following methods return a boolean value. 

- `A.issubset(B)` :  $A \subset B$
- `A.issuperset(B)` : $A \supset B$
- `elem in A` : $x \in A$

The following methods update the instance which is applied to. (`elem` is a
hashable object.)

- `add(elem)` : adds an element to the set.
- `remove(elem)` : removes the element from the set.
  This raises KeyError if the element is not contained in the set.
- `discard(elem)` : removes element elem from the set. 
  If the element does not belong to the set, nothing happens.
- `pop()` : removes an arbitrary element from the set and returns it.
  This Raises KeyError if the set is empty.
- `clear()` removes all elements from the set.

The following methods return a new instance and do not change the instance
which is applied to.

- `A.intersection(B)` : $A \cap B$
- `A.union(B)` : $A \cup B$
- `A.difference(B)` : $A \setminus B$,
- `A - B ` : same as above

