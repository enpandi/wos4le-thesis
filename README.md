This repository contains code associated with my undergraduate thesis, "Walk on Spheres for Linear Elasticity".
The thesis document can be found [here](https://apps.cs.utexas.edu/apps/tech-reports/206121).

To build and run:
```
(mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8) && build/wos4le cow.off
```
`cmake` clones the dependencies (libigl, polscope) so it may take a while.

cow.off from [libigl tutorial data](https://github.com/libigl/libigl-tutorial-data),
sphere.off from [here](https://github.com/noamaig/euclidean_orbifolds).
