// Codebase modules
mod simulation;
mod physics;
mod visualisation;
mod extfields;
mod compute;

use crate::compute::ComputeCore;
use crate::simulation::{VortexSimulation, run_parameter_study, SimulationResult};

// External crates
use clap::{Arg, Command};
use std::{default, fs::create_dir_all};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Vortex Cores")
        .version("0.1")
        .about("Simulates quantum vortex dynamics in superfluid helium")
        .subcommand(
            Command::new("single")
                .about("Run a single simulation with specified parameters")
                .arg(
                    Arg::new("gpu")
                        .short('g')
                        .long("gpu")
                        .help("Use GPU for computation")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    Arg::new("list_gpus")
                        .long("list-gpus")
                        .help("List available GPUs and exit")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    Arg::new("select_gpu")
                        .long("select-gpu")
                        .help("Select a specific GPU by name fragment (case-insensitive)")
                        .value_name("NAME")
                )
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
                    Arg::new("waves")
                        .short('w')
                        .long("waves")
                        .value_name("AMPLITUDE")
                        .help("Kelvin wave amplitude factor (0.0-1.0)")
                        .default_value("0.1")
                )
                .arg(
                    Arg::new("ext_field")
                        .short('e')
                        .long("ext-field")
                        .value_name("TYPE")
                        .help("External field type (none, rotation, uniform, oscillatory, counterflow)")
                        .default_value("none")
                )
                .arg(
                    Arg::new("ext_value")
                        .long("ext-value")
                        .value_name("VALUE")
                        .help("External field value (format depends on field type)")
                        .default_value("0.0,0.0,0.0")
                )
                .arg(
                    Arg::new("ext_center")
                        .long("ext-center")
                        .value_name("CENTER")
                        .help("Center point for rotation field (x,y,z)")
                        .default_value("0.0,0.0,0.0")
                )
                .arg(
                    Arg::new("ext_freq")
                        .long("ext-freq")
                        .value_name("FREQ")
                        .help("Frequency for oscillatory flow (Hz)")
                        .default_value("1.0")
                )
                .arg(
                    Arg::new("ext_phase")
                        .long("ext-phase")
                        .value_name("PHASE")
                        .help("Phase for oscillatory flow (radians)")
                        .default_value("0.0")
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
                    Arg::new("gpu")
                        .short('g')
                        .long("gpu")
                        .help("Use GPU for computation")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    Arg::new("list_gpus")
                        .long("list-gpus")
                        .help("List available GPUs and exit")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    Arg::new("select_gpu")
                        .long("select-gpu")
                        .help("Select a specific GPU by name fragment (case-insensitive)")
                        .value_name("NAME")
                )
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
                    Arg::new("waves")
                        .short('w')
                        .long("waves")
                        .value_name("AMPLITUDE")
                        .help("Kelvin wave amplitude factor (0.0-1.0)")
                        .default_value("0.1")
                )
                .arg(
                    Arg::new("ext_field")
                        .short('e')
                        .long("ext-field")
                        .value_name("TYPE")
                        .help("External field type (none, rotation, uniform, oscillatory, counterflow)")
                        .default_value("none")
                )
                .arg(
                    Arg::new("ext_value")
                        .long("ext-value")
                        .value_name("VALUE")
                        .help("External field value (format depends on field type)")
                        .default_value("0.0,0.0,0.0")
                )
                .arg(
                    Arg::new("ext_center")
                        .long("ext-center")
                        .value_name("CENTER")
                        .help("Center point for rotation field (x,y,z)")
                        .default_value("0.0,0.0,0.0")
                )
                .arg(
                    Arg::new("ext_freq")
                        .long("ext-freq")
                        .value_name("FREQ")
                        .help("Frequency for oscillatory flow (Hz)")
                        .default_value("1.0")
                )
                .arg(
                    Arg::new("ext_phase")
                        .long("ext-phase")
                        .value_name("PHASE")
                        .help("Phase for oscillatory flow (radians)")
                        .default_value("0.0")
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
        // If list-gpus flag is set, list available GPUs and exit
        if matches.get_flag("list_gpus") {
            println!("Listing available GPUs...");
            let gpus = pollster::block_on(async { ComputeCore::list_available_gpus().await });
            
            if gpus.is_empty() {
                println!("No compatible GPUs found!");
            } else {
                println!("Available GPUs:");
                for (i, (name, device_type, compatibility)) in gpus.iter().enumerate() {
                    println!("  {}: {} ({}) - {}", i, name, 
                        match device_type {
                            wgpu::DeviceType::DiscreteGpu => "Discrete",
                            wgpu::DeviceType::IntegratedGpu => "Integrated",
                            wgpu::DeviceType::Cpu => "CPU",
                            wgpu::DeviceType::Other => "Other",
                            _ => "Unknown",
                        },
                        compatibility
                    );
                }
            }
            return Ok(());
        }
        let radius = matches.get_one::<String>("radius").unwrap().parse::<f64>()?;
        let height = matches.get_one::<String>("height").unwrap().parse::<f64>()?;
        let temperature = matches.get_one::<String>("temperature").unwrap().parse::<f64>()?;
        let wave_amplitude = matches.get_one::<String>("waves").unwrap().parse::<f64>()?;
        let steps = matches.get_one::<String>("steps").unwrap().parse::<usize>()?;
        let output = matches.get_one::<String>("output").unwrap();
        let use_gpu = matches.get_flag("gpu");
        let selected_gpu = matches.get_one::<String>("select_gpu").map(|s| s.as_str());
        
        // Process external field parameters
        let external_field = simulation::parse_external_field(
            matches.get_one::<String>("ext_field").unwrap(),
            matches.get_one::<String>("ext_value").unwrap(),
            matches.get_one::<String>("ext_center").unwrap(),
            matches.get_one::<String>("ext_freq").unwrap().parse::<f64>()?,
            matches.get_one::<String>("ext_phase").unwrap().parse::<f64>()?,
        )?;

        println!("Running single simulation with parameters:");
        println!("  Radius: {} cm", radius);
        println!("  Height: {} cm", height);
        println!("  Temperature: {} K", temperature);
        println!("  Steps: {}", steps);
        if let Some(field) = &external_field {
            println!("  External field: {:?}", field);
        }
        println!("  Using GPU: {}", use_gpu);

        // Create simulation and optionally initialize GPU
        let mut sim = if use_gpu {
            println!("Initializing GPU...");
            pollster::block_on(async {
                if let Some(gpu_name) = selected_gpu {
                    let compute_core = ComputeCore::new_with_device_preference(Some(gpu_name)).await;
                    VortexSimulation::new_with_compute_core(radius, height, temperature, compute_core)
                } else {
                    VortexSimulation::new_with_gpu(radius, height, temperature).await
                }
            })
        } else {
            VortexSimulation::new(radius, height, temperature)
        };
        
        // Set external field if specified
        if let Some(field) = external_field {
            sim.external_field = Some(field);
        }
        
        if wave_amplitude > 0.0 {
            sim.add_kelvin_waves(wave_amplitude * sim.radius, 3.0);
        }
        
        sim.run(steps);
        sim.save_results(output);
        
        println!("Simulation complete! Results saved to {}", output);
    }
    // Handle parameter study
    else if let Some(matches) = matches.subcommand_matches("study") {
        // If list-gpus flag is set, list available GPUs and exit
        if matches.get_flag("list_gpus") {
            println!("Listing available GPUs...");
            let gpus = pollster::block_on(async { ComputeCore::list_available_gpus().await });
            
            if gpus.is_empty() {
                println!("No compatible GPUs found!");
            } else {
                println!("Available GPUs:");
                for (i, (name, device_type, compatibility)) in gpus.iter().enumerate() {
                    println!("  {}: {} ({}) - {}", i, name, 
                        match device_type {
                            wgpu::DeviceType::DiscreteGpu => "Discrete",
                            wgpu::DeviceType::IntegratedGpu => "Integrated",
                            wgpu::DeviceType::Cpu => "CPU",
                            wgpu::DeviceType::Other => "Other",
                            _ => "Unknown",
                        },
                        compatibility
                    );
                }
            }
            return Ok(());
        }

        let radius_min = matches.get_one::<String>("radius_min").unwrap().parse::<f64>()?;
        let radius_max = matches.get_one::<String>("radius_max").unwrap().parse::<f64>()?;
        let radius_steps = matches.get_one::<String>("radius_steps").unwrap().parse::<usize>()?;
        let height = matches.get_one::<String>("height").unwrap().parse::<f64>()?;
        let temp_min = matches.get_one::<String>("temp_min").unwrap().parse::<f64>()?;
        let temp_max = matches.get_one::<String>("temp_max").unwrap().parse::<f64>()?;
        let temp_steps = matches.get_one::<String>("temp_steps").unwrap().parse::<usize>()?;
        let wave_amplitude = matches.get_one::<String>("waves").unwrap().parse::<f64>()?;
        let steps = matches.get_one::<String>("sim_steps").unwrap().parse::<usize>()?;
        let output_dir = matches.get_one::<String>("output_dir").unwrap();
        let use_gpu = matches.get_flag("gpu");
        let selected_gpu = matches.get_one::<String>("select_gpu").map(|s| s.as_str());
        
        // Process external field parameters
        let external_field = simulation::parse_external_field(
            matches.get_one::<String>("ext_field").unwrap(),
            matches.get_one::<String>("ext_value").unwrap(),
            matches.get_one::<String>("ext_center").unwrap(),
            matches.get_one::<String>("ext_freq").unwrap().parse::<f64>()?,
            matches.get_one::<String>("ext_phase").unwrap().parse::<f64>()?,
        )?;
        
        // Create output directory if it doesn't exist
        if !Path::new(output_dir).exists() {
            create_dir_all(output_dir)?;
        }
        
        println!("Running parameter study with:");
        println!("  Radius range: {} to {} cm ({} steps)", radius_min, radius_max, radius_steps);
        println!("  Height: {} cm", height);
        println!("  Temperature range: {} to {} K ({} steps)", temp_min, temp_max, temp_steps);
        println!("  Kelvin wave amplitude: {}", wave_amplitude);
        println!("  Simulation steps: {}", steps);
        if let Some(field) = &external_field {
            println!("  External field: {:?}", field);
        }
        println!("  Using GPU: {}", use_gpu);
        println!("  Output directory: {}", output_dir);
        
        // Initialize GPU compute core if requested
        let compute_core = if use_gpu {
            println!("Initializing GPU...");
            Some(pollster::block_on(async {
                if let Some(gpu_name) = selected_gpu {
                    ComputeCore::new_with_device_preference(Some(gpu_name)).await
                } else {
                    ComputeCore::new().await
                }
            }))
        } else {
            None
        };
        
        // Run the parameter study with GPU if selected
        let results = run_parameter_study(
            height / 2.0, // base_radius (not used directly in sweep)
            height,
            (temp_min + temp_max) / 2.0, // base_temperature (not used directly in sweep)
            (radius_min, radius_max, radius_steps),
            (temp_min, temp_max, temp_steps),
            wave_amplitude,
            steps,
            output_dir,
            compute_core,
            external_field,
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
