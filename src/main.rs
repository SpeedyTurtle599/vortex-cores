// Codebase modules
mod simulation;
mod physics;
mod visualisation;
mod extfields;

// External crates
use clap::{Arg, Command};
use std::fs::create_dir_all;
use std::path::Path;
use crate::simulation::{VortexSimulation, run_parameter_study, SimulationResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Vortex Cores")
        .version("0.1")
        .about("Simulates quantum vortex dynamics in superfluid helium")
        .subcommand(
            Command::new("single")
                .about("Run a single simulation with specified parameters")
                .arg(
                    Arg::new("radius")
                        .short('r')
                        .long("radius")
                        .value_name("RADIUS")
                        .help("Cylinder radius in cm")
                        .default_value("0.5"),
                )
                .arg(
                    Arg::new("height")
                        .short('H')
                        .long("height")
                        .value_name("HEIGHT")
                        .help("Cylinder height in cm")
                        .default_value("1.0"),
                )
                .arg(
                    Arg::new("temperature")
                        .short('t')
                        .long("temp")
                        .value_name("TEMP")
                        .help("Temperature in Kelvin")
                        .default_value("1.5"),
                )
                .arg(
                    Arg::new("steps")
                        .short('s')
                        .long("steps")
                        .value_name("STEPS")
                        .help("Number of simulation steps")
                        .default_value("1000"),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output file name")
                        .default_value("output.vtk"),
                )
        )
        .subcommand(
            Command::new("study")
                .about("Run a parameter study with ranges of values")
                .arg(
                    Arg::new("radius_min")
                        .long("rmin")
                        .value_name("RADIUS_MIN")
                        .help("Minimum cylinder radius in cm")
                        .default_value("0.2"),
                )
                .arg(
                    Arg::new("radius_max")
                        .long("rmax")
                        .value_name("RADIUS_MAX")
                        .help("Maximum cylinder radius in cm")
                        .default_value("1.0"),
                )
                .arg(
                    Arg::new("radius_steps")
                        .long("rsteps")
                        .value_name("RADIUS_STEPS")
                        .help("Number of radius steps")
                        .default_value("3"),
                )
                .arg(
                    Arg::new("height")
                        .short('h')
                        .long("height")
                        .value_name("HEIGHT")
                        .help("Cylinder height in cm")
                        .default_value("1.0"),
                )
                .arg(
                    Arg::new("temp_min")
                        .long("tmin")
                        .value_name("TEMP_MIN")
                        .help("Minimum temperature in Kelvin")
                        .default_value("1.0"),
                )
                .arg(
                    Arg::new("temp_max")
                        .long("tmax")
                        .value_name("TEMP_MAX")
                        .help("Maximum temperature in Kelvin")
                        .default_value("2.1"),
                )
                .arg(
                    Arg::new("temp_steps")
                        .long("tsteps")
                        .value_name("TEMP_STEPS")
                        .help("Number of temperature steps")
                        .default_value("5"),
                )
                .arg(
                    Arg::new("sim_steps")
                        .short('s')
                        .long("steps")
                        .value_name("SIM_STEPS")
                        .help("Number of simulation steps")
                        .default_value("500"),
                )
                .arg(
                    Arg::new("output_dir")
                        .short('o')
                        .long("output")
                        .value_name("DIR")
                        .help("Output directory for study results")
                        .default_value("study_results"),
                )
        )
        .get_matches();

    // Handle single simulation
    if let Some(matches) = matches.subcommand_matches("single") {
        let radius = matches.get_one::<String>("radius").unwrap().parse::<f64>()?;
        let height = matches.get_one::<String>("height").unwrap().parse::<f64>()?;
        let temperature = matches.get_one::<String>("temperature").unwrap().parse::<f64>()?;
        let steps = matches.get_one::<String>("steps").unwrap().parse::<usize>()?;
        let output = matches.get_one::<String>("output").unwrap();

        println!("Running single simulation with parameters:");
        println!("  Radius: {} cm", radius);
        println!("  Height: {} cm", height);
        println!("  Temperature: {} K", temperature);
        println!("  Steps: {}", steps);

        let mut sim = VortexSimulation::new(radius, height, temperature);
        sim.run(steps);
        sim.save_results(output);
        
        println!("Simulation complete! Results saved to {}", output);
    }
    // Handle parameter study
    else if let Some(matches) = matches.subcommand_matches("study") {
        let radius_min = matches.get_one::<String>("radius_min").unwrap().parse::<f64>()?;
        let radius_max = matches.get_one::<String>("radius_max").unwrap().parse::<f64>()?;
        let radius_steps = matches.get_one::<String>("radius_steps").unwrap().parse::<usize>()?;
        let height = matches.get_one::<String>("height").unwrap().parse::<f64>()?;
        let temp_min = matches.get_one::<String>("temp_min").unwrap().parse::<f64>()?;
        let temp_max = matches.get_one::<String>("temp_max").unwrap().parse::<f64>()?;
        let temp_steps = matches.get_one::<String>("temp_steps").unwrap().parse::<usize>()?;
        let steps = matches.get_one::<String>("sim_steps").unwrap().parse::<usize>()?;
        let output_dir = matches.get_one::<String>("output_dir").unwrap();
        
        // Create output directory if it doesn't exist
        if !Path::new(output_dir).exists() {
            create_dir_all(output_dir)?;
        }
        
        println!("Running parameter study with:");
        println!("  Radius range: {} to {} cm ({} steps)", radius_min, radius_max, radius_steps);
        println!("  Height: {} cm", height);
        println!("  Temperature range: {} to {} K ({} steps)", temp_min, temp_max, temp_steps);
        println!("  Simulation steps: {}", steps);
        println!("  Output directory: {}", output_dir);
        
        // Run the parameter study
        let results = run_parameter_study(
            height / 2.0, // base_radius (not used directly in sweep)
            height,
            (temp_min + temp_max) / 2.0, // base_temperature (not used directly in sweep)
            (radius_min, radius_max, radius_steps),
            (temp_min, temp_max, temp_steps),
            steps,
            output_dir,
        );
        
        // Save summary results
        let summary_file = format!("{}/summary.json", output_dir);
        let file = std::fs::File::create(&summary_file)?;
        serde_json::to_writer_pretty(file, &results)?;
        
        println!("Parameter study complete! Summary saved to {}", summary_file);
    }
    // Default behavior if no subcommand is provided
    else {
        println!("Please specify a subcommand: 'single' or 'study'");
        println!("Use --help for more information");
    }

    Ok(())
}