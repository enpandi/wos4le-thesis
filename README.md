This repository contains code associated with my undergraduate thesis, "Walk on Spheres for Linear Elasticity".
The thesis document can be found [here](https://apps.cs.utexas.edu/apps/tech-reports/206121).

To build and run:
```
(mkdir -p build && cd build && cmake .. && make -j8) && build/wos4le
```
`cmake` clones the dependencies (libigl, polscope) so it may take a while.
