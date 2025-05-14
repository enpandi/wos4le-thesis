files/code associated with my undergraduate thesis, "Walk on Spheres for Linear Elasticity"

to build and run:
```
(mkdir -p build && cd build && cmake .. && make -j8) && build/wos4le
```
`cmake` clones the dependencies (libigl, polscope) so it will take a while
