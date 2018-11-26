# pykifmm2d: Kernel Independent FMM in Two Dimensions
 
Provide a pure python implementation of the Kernel Independent Fast-Multipole method in two dimensions, largely based off of the algorithm presented in the paper [A kernel-independent adaptive fast multipole
algorithm in two and three dimensions](https://www.mrl.nyu.edu/~harper/kifmm3d/documentation/papers/fmm.pdf). The algorithm differs somewhat, however, since its based on a level-restricted quadtree instead of a general quadtree.

The primary purpose of this project was initially for me to understand what an FMM was doing, how it was implemented, and if it was possible to write a relatively efficient version of one in python (+numba acceleration). Nevertheless, I'm not aware of any other easily available / installable KIFMM implementations in 2D so this may be of some use to others. I have no plans to extend this to three dimensions, where well-optimized and scalable codes already exist, e.g [PVFMM](https://github.com/dmalhotra/pvfmm).

## What's here:
This project is in an early state. For now, the only things implemented are:

1. Trees
	1. There are two trees (well, three, really). I've rewritten the tree a few times:
		a. The first tree I wrote is in tree.py; this isn't in use any more.
		b. The second is in tree2.py, this is a pretty fast for small trees and large cutoffs but slows down for larger trees and small cutoffs.  Its a simple implemenation and is currently used by the FMM.
		c. The third tree is in tree3.py, and is well optimized and quite fast.
	2. In the near future, the FMM will be moved over to using the optimized tree from tree3.py.  There are still a few things to implement in tree3.py and a few things that I need to think about before doing this.
2. A basic FMM code for computing the sum of G (and only G, not derivatives...) from a set of sources to itself, ignoring the self-interaction
	1. This code does not currently have support for either S-lists or X-lists
	2. For S-lists, all neighbor cells at the same level are evaluated directly. There are situations, particularly in heavily nested trees, where this means that there may be very large direct interactions. This could considerably impact performance. In the tree structure in tree3.py, I have made a slightly different decision: leaves are refined if they're more than 1 level in depth apart from any colleagues.  This prevents S-lists that are worse than a 4-to-1 ratio, and (I'm pretty sure) leads to no infinitely shitty situations, either in refinement or evaluation.  I think this is a worthwhile compromise to make for the code simplifications it gives.
	3. For X-lists, the leaf is 'refined' into 'fake children', which allows significant simplification in implementation, however may have some effect on performance, both for tree formation and in the actual FMM. I probably don't intend to ever fix this, unless it proves to be a serious issue in terms of performance.
	4. This basic FMM will be largely left in-tact, as an unoptimized, simple to understand code, even if I ultimately write more optimized versions. I hope that others can use this code to learn from, even if its not the fastest code out there.
3. A few examples: for basic tree formation and Laplace kernels. The Laplace example requires numba and numexpr to run correctly.

## What's coming:

1. Properly implemented S-list interactions
	a. Scratch that: a new tree that makes treating S-list interactions naively not a big deal.
2. Planned FMMs (for faster evaluation of repeated FMMs)
3. Optimized versions
4. A version robust to rapidly decaying Greens functions (e.g. modified Helmholtz with large k)
5. More examples!
	1. Heat kernels
	2. Examples comparing to the [FMMLIB2D](https://github.com/zgimbutas/fmmlib2d) code exposed through my python wrapper [pyfmmlib2d](https://github.com/dbstein/pyfmmlib2d).
	3. Stokes/Brinkman, once I figure out how to implement them.

## Installation
Install by navigating to the package route and typing:
```bash
pip install .
```

### What's required:

1. Not much: a base python installation and numpy/scipy, and numba
2. Some examples require numexpr and matplotlib
3. For optimal performance, I recommend using the [Intel Distribution for Python](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda)

