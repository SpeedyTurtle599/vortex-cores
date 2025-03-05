mod simulation;
mod physics;
mod visualisation;

fn main() {
    println!("Superfluid Helium Vortex Core Simulation");
    
    // Create simulation parameters
    let cylinder_radius = 1.0; // cm
    let cylinder_height = 2.0; // cm
    let temperature = 1.5;     // Kelvin
    let simulation_steps = 1000;
    
    // Initialize simulation
    let mut sim = simulation::VortexSimulation::new(
        cylinder_radius,
        cylinder_height,
        temperature,
    );
    
    // Run simulation
    sim.run(simulation_steps);
    
    // Output results
    sim.save_results("vortex_tangle.vtk");
    
    println!("Simulation complete!");
}