### Introduction
This codebase simulates the behavior of superfluid helium vortex cores. MORE DESCRIPTION?

### Usage

Single run, uniform x-axis external flow, velocity 0.1 cm/s:
```
cargo run -- single --gpu --radius 0.5 --height 1.0 --temp 1.5 --steps 1000 \
  --ext-field rotation --ext-value 0,0,1.0 --ext-center 0,0,0.5
```

Single run, rotation field with z-axis rotation, centered at (0, 0, 0.5), angular velocity 1.0 rad/s:
```
cargo run -- single --gpu --radius 0.5 --height 1.0 --temp 1.5 --steps 1000 \
  --ext-field rotation --ext-value 0,0,1.0 --ext-center 0,0,0.5
```

Parameter study:
```
cargo run -- study --rmin 0.2 --rmax 1.0 --rsteps 4 --tmin 1.0 --tmax 2.1 --tsteps 5 --steps 500 --output study_results
```

### To Do
- Update Cargo.toml to add webGPU dependencies for GPU acceleration
- Debug apparent hang after 2 reconnections
- Add more detailed description of the codebase and underlying physics