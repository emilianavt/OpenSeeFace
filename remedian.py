"""
Watch over a stream of numbers, incrementally learning their median.

Implemented via nested lists. New numbers are added to `lst[i]` and
when it fills up, it posts its median to `lst[i+1]`. Wen `lst[i+1]`
fills up, it posts the medians of its medians to `lst[i+2]`. Etc.
When a remedian is queried for the current median, it returns the
median of the last list with any numbers.

This approach is quite space efficient . E.g. four nested lists,
each with 64 items, require memory for 4*64 items yet can hold the
median of the median of the median of the median of over 17 million
numbers.

Example usage:

        z=remedian()
        for i in range(1000):
          z + i
          if not i % 100:
            print(i, z.median())

Based on  [The Remedian:A Robust Averaging Method for Large Data
Sets](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/Remedian.pdf).
by Peter J. Rousseeuw and Gilbert W. Bassett Jr.  Journal of the
American Statistical Association March 1990, Vol. 85, No. 409,
Theory and Methods

The code [remedianeg.py](remedianeg.py) compares this rig to just
using Python's built-in sort then reporing the middle number.
Assuming lists of length 64 and use of pypy3:

- Remedian is getting nearly as fast (within 20%) as raw sort after 500 items;
- While at the same time, avoids having to store all the numbers in RAM;
- Further, remedian's computed median is within 1% (or less) of the medians found via Python's sort.

_____

## Programmer's Guide

"""

# If `ordered` is `False`, do not sort `lst`
def median(lst,ordered=False):
  assert lst,"median needs a non-empty list"
  n  = len(lst)
  p  = q  = n//2
  if n < 3:
    p,q = 0, n-1
  else:
    lst = lst if ordered else sorted(lst)
    if not n % 2: # for even-length lists, use mean of mid 2 nums
      q = p -1
  return lst[p] if p==q else (lst[p]+lst[q])/2

class remedian:

  # Initialization
  def __init__(i,inits=[], k=64, # after some experimentation, 64 works ok
               about = None):
    i.all,i.k = [],k
    i.more,i._median=None,None
    [i + x for x in inits]

  # When full, push the median of current values to next list, then reset.
  def __add__(i,x):
    i._median = None
    i.all.append(x)
    if len(i.all) == i.k:
      i.more = i.more or remedian(k=i.k)
      i.more + i._medianPrim(i.all)
      i.all = []  # reset

  #  If there is a next list, ask its median. Else, work it out locally.
  def median(i):
    return i.more.median() if i.more else i._medianPrim(i.all)

  # Only recompute median if we do not know it already.
  def _medianPrim(i,all):
    if i._median == None:
      i._median = median(all,ordered=False)
    return i._median
