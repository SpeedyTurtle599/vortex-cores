### Introduction
This codebase simulates the behavior of superfluid helium vortex cores. MORE DESCRIPTION?

### Usage

Single run:
```
cargo run -- single --radius 0.5 --height 1.0 --temp 1.5 --steps 1000 --output results.vtk
```

Parameter study:
```
cargo run -- study --rmin 0.2 --rmax 1.0 --rsteps 4 --tmin 1.0 --tmax 2.1 --tsteps 5 --steps 500 --output study_results
```