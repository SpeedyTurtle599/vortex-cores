mod simulation;
mod physics;
mod visualisation;
mod extfields;

use clap::Parser;
use extfields::ExternalField;
use nalgebra::Vector3;

/// Superfluid Helium Vortex Core Simulation
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Cylinder radius in cm
    #[clap(short, long, default_value = "1.0")]
    radius: f64,
    
    /// Cylinder height in cm
    #[clap(short, long, default_value = "2.0")]
    height: f64,
    
    /// Temperature in Kelvin
    #[clap(short, long, default_value = "1.5")]
    temperature: f64,
    
    /// Number of simulation steps
    #[clap(short, long, default_value = "1000")]
    steps: usize,
    
    /// Output filename for results
    #[clap(short, long, default_value = "vortex_tangle.vtk")]
    output: String,
    
    /// Enable rotation (rad/s)
    #[clap(long)]
    rotation: Option<f64>,
    
    /// Enable uniform flow in z-direction (cm/s)
    #[clap(long)]
    flow: Option<f64>,
    
    /// Enable oscillatory flow (cm/s)
    #[clap(long)]
    oscillation: Option<f64>,
    
    /// Oscillation frequency (Hz)
    #[clap(long, default_value = "10.0")]
    frequency: f64,
    
    /// Enable Kelvin wave initialization
    #[clap(long)]
    kelvin: bool,
}

fn main() {
    let args = Args::parse();
    
    println!("Superfluid Helium Vortex Core Simulation");
    println!("----------------------------------------");
    println!("Cylinder: radius = {} cm, height = {} cm", args.radius, args.height);
    println!("Temperature: {} K", args.temperature);
    println!("Steps: {}", args.steps);
    println!("Kelvin wave initialization: {}", args.kelvin);
    
    // Initialize simulation
    let mut sim = simulation::VortexSimulation::new(
        args.radius,
        args.height,
        args.temperature,
    );
    
    // Set external field based on command-line args
    if let Some(omega) = args.rotation {
        println!("External rotation: {} rad/s", omega);
        sim.set_external_field(ExternalField::Rotation {
            angular_velocity: Vector3::new(0.0, 0.0, omega),
            center: Vector3::new(0.0, 0.0, args.height/2.0),
        });
    }
    else if let Some(velocity) = args.flow {
        println!("Uniform flow: {} cm/s in z-direction", velocity);
        sim.set_external_field(ExternalField::UniformFlow {
            velocity: Vector3::new(0.0, 0.0, velocity),
        });
    }
    else if let Some(amplitude) = args.oscillation {
        println!("Oscillatory flow: {} cm/s at {} Hz", amplitude, args.frequency);
        sim.set_external_field(ExternalField::OscillatoryFlow {
            amplitude: Vector3::new(0.0, 0.0, amplitude),
            frequency: args.frequency,
            phase: 0.0,
        });
    }
    
    // Run simulation
    sim.run(args.steps);
    
    // Output results
    sim.save_results(&args.output);
    
    println!("Simulation complete!");
}