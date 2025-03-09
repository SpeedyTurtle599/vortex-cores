// Codebase modules
mod simulation;
mod physics;
mod extfields;
mod compute;

use crate::compute::ComputeCore;
use crate::simulation::{VortexSimulation, run_parameter_study};

// External crates
use clap::{Arg, Command};
use std::fs::create_dir_all;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Vortex Cores")
        .version("0.1")
        .about("Simulates quantum vortex dynamics in superfluid helium")
        .subcommand(
            Command::new("single")
                .about("Run a single simulation")
                .arg(Arg::new("gpu")
                    .long("gpu")
                    .help("Use GPU acceleration")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("list_gpus")
                    .long("list-gpus")
                    .help("List available GPUs")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("select_gpu")
                    .long("select-gpu")
                    .help("Select specific GPU by name fragment")
                    .value_name("NAME")
                    .value_parser(clap::value_parser!(String)))
                .arg(Arg::new("steps")
                    .short('s')
                    .long("steps")
                    .help("Number of simulation steps")
                    .value_name("STEPS")
                    .value_parser(clap::value_parser!(String))
                    .default_value("1000"))
                .arg(Arg::new("output")
                    .short('o')
                    .long("output")
                    .help("Output file path")
                    .value_name("FILE")
                    .value_parser(clap::value_parser!(String))
                    .default_value("output.vtk"))
                .arg(Arg::new("time_series")
                    .long("time-series")
                    .value_name("INTERVAL")
                    .help("Save time series VTK files every N steps")
                    .num_args(1))
                .arg(Arg::new("time_series_name")
                    .long("series-name")
                    .value_name("NAME")
                    .help("Base name for time series files")
                    .default_value("vortex_time_series")
                    .num_args(1))
                .arg(Arg::new("load_checkpoint")
                    .short('l')
                    .long("load")
                    .help("Load from checkpoint file")
                    .value_name("FILE")
                    .value_parser(clap::value_parser!(String)))
                    .arg(Arg::new("radius")
                    .long("radius")
                    .short('r')
                    .help("Container radius in cm")
                    .value_name("RADIUS")
                    .value_parser(clap::value_parser!(String))
                    .default_value("1.0"))
                .arg(Arg::new("height")
                    .long("height")
                    .short('H')
                    .help("Container height in cm")
                    .value_name("HEIGHT")
                    .value_parser(clap::value_parser!(String))
                    .default_value("2.0"))
                .arg(Arg::new("temperature")
                    .long("temp")
                    .short('t')
                    .help("Temperature in Kelvin")
                    .value_name("TEMP")
                    .value_parser(clap::value_parser!(String))
                    .default_value("1.5"))
                .arg(Arg::new("ext_field")
                    .short('f')
                    .long("ext-field")
                    .help("External field type: none, rotation, uniform, oscillatory, counterflow")
                    .value_name("TYPE")
                    .value_parser(clap::value_parser!(String))
                    .default_value("none"))
                .arg(Arg::new("field_value")
                    .short('v')
                    .long("ext-value")
                    .help("External field value (e.g. '0,0,1' for rotation)")
                    .value_name("VALUE")
                    .value_parser(clap::value_parser!(String))
                    .default_value("0,0,0"))
                .arg(Arg::new("field_center")
                    .short('c')
                    .long("ext-center")
                    .help("Field center for rotation (e.g. '0,0,0')")
                    .value_name("CENTER")
                    .value_parser(clap::value_parser!(String))
                    .default_value("0,0,0"))
                .arg(Arg::new("frequency")
                    .long("ext-freq")
                    .help("Frequency for oscillatory field (Hz)")
                    .value_name("FREQ")
                    .value_parser(clap::value_parser!(f64))
                    .default_value("1.0"))
                .arg(Arg::new("phase")
                    .long("ext-phase")
                    .help("Phase for oscillatory field (radians)")
                    .value_name("PHASE")
                    .value_parser(clap::value_parser!(f64))
                    .default_value("0.0"))
                .arg(Arg::new("waves")
                    .short('w')
                    .long("waves")
                    .help("Add Kelvin waves with given amplitude")
                    .value_name("AMPLITUDE")
                    .value_parser(clap::value_parser!(String))
                    .default_value("0.0"))
        )
        .subcommand(
            Command::new("resume")
                .about("Resume from checkpoint")
                .arg(Arg::new("gpu")
                    .long("gpu")
                    .help("Use GPU acceleration")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("select_gpu")
                    .long("select-gpu")
                    .help("Select specific GPU by name fragment")
                    .value_name("NAME"))
                .arg(Arg::new("checkpoint")
                    .required(true)
                    .help("Checkpoint file to resume from")
                    .value_name("FILE"))
                .arg(Arg::new("steps")
                    .short('s')
                    .long("steps")
                    .help("Number of additional simulation steps")
                    .value_name("STEPS")
                    .default_value("1000"))
                .arg(Arg::new("output")
                    .short('o')
                    .long("output")
                    .help("Output file path")
                    .value_name("FILE")
                    .default_value("continued.vtk"))
        )
        .subcommand(
            Command::new("study")
                .about("Run parameter study")
                .arg(Arg::new("gpu")
                    .long("gpu")
                    .help("Use GPU acceleration")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("list_gpus")
                    .long("list-gpus")
                    .help("List available GPUs")
                    .action(clap::ArgAction::SetTrue))
                .arg(Arg::new("select_gpu")
                    .long("select-gpu")
                    .help("Select specific GPU by name fragment")
                    .value_name("NAME"))
                .arg(Arg::new("radius_min")
                    .long("rmin")
                    .help("Minimum radius (cm)")
                    .value_name("RMIN")
                    .default_value("0.5"))
                .arg(Arg::new("radius_max")
                    .long("rmax")
                    .help("Maximum radius (cm)")
                    .value_name("RMAX")
                    .default_value("2.0"))
                .arg(Arg::new("radius_steps")
                    .long("rsteps")
                    .help("Number of radius values to test")
                    .value_name("RSTEPS")
                    .default_value("4"))
                .arg(Arg::new("temperature_min")
                    .long("tmin")
                    .help("Minimum temperature (K)")
                    .value_name("TMIN")
                    .default_value("1.0"))
                .arg(Arg::new("temperature_max")
                    .long("tmax")
                    .help("Maximum temperature (K)")
                    .value_name("TMAX")
                    .default_value("2.0"))
                .arg(Arg::new("temperature_steps")
                    .long("tsteps")
                    .help("Number of temperature values to test")
                    .value_name("TSTEPS")
                    .default_value("3"))
                .arg(Arg::new("output_dir")
                    .short('o')
                    .long("output-dir")
                    .help("Output directory")
                    .value_name("DIR")
                    .default_value("study_results"))
                .arg(Arg::new("steps")
                    .short('s')
                    .long("steps")
                    .help("Steps per simulation")
                    .value_name("STEPS")
                    .default_value("500"))
                .arg(Arg::new("waves")
                    .short('w')
                    .long("waves")
                    .help("Add Kelvin waves with given amplitude (0.0 for none)")
                    .value_name("AMPLITUDE")
                    .default_value("0.1"))
                .arg(Arg::new("ext_field")
                    .short('f')
                    .long("field")
                    .help("External field type: none, rotation, uniform, oscillatory, counterflow")
                    .value_name("TYPE")
                    .default_value("none"))
                .arg(Arg::new("field_value")
                    .short('v')
                    .long("value")
                    .help("External field value (e.g. '0,0,1' for rotation)")
                    .value_name("VALUE")
                    .default_value("0,0,0"))
        )
        .get_matches();

    // Handle single simulation
    if let Some(matches) = matches.subcommand_matches("single") {
        // If list-gpus flag is set, list available GPUs and exit
        if matches.get_flag("list_gpus") {
            println!("Available GPUs:");
            let gpu_list = pollster::block_on(ComputeCore::list_available_gpus());
            for (i, (name, device_type, compatibility)) in gpu_list.iter().enumerate() {
                println!("  {}. {} ({}) - {}", i+1, name, device_type_to_string(*device_type), compatibility);
            }
            return Ok(());
        }
        
        let use_gpu = matches.get_flag("gpu");
        let selected_gpu = matches.get_one::<String>("select_gpu").map(|s| s.as_str());
        let steps = matches.get_one::<String>("steps").unwrap().parse::<usize>()?;
        let output = matches.get_one::<String>("output").unwrap();
        
        // Initialize GPU compute core if needed
        let compute_core = if use_gpu {
            simulation::log_message(&format!("Initializing GPU compute core..."));
            let core = pollster::block_on(ComputeCore::new_with_device_preference(selected_gpu));
            Some(core)
        } else {
            None
        };
        
        // Check if we should load from checkpoint or create a new simulation
        let mut sim = if let Some(checkpoint_file) = matches.get_one::<String>("load_checkpoint") {
            simulation::log_message(&format!("Loading simulation from checkpoint: {}", checkpoint_file));
            let mut sim = VortexSimulation::load_checkpoint(checkpoint_file)?;
            
            // Set compute core if using GPU
            if let Some(core) = compute_core {
                sim.set_compute_core(core);
            }
            
            sim
        } else {
            // Create a new simulation
            let cylinder_radius = matches.get_one::<String>("radius").unwrap().parse::<f64>()?;
            let cylinder_height = matches.get_one::<String>("height").unwrap().parse::<f64>()?;
            let temperature = matches.get_one::<String>("temperature").unwrap().parse::<f64>()?;
            
            simulation::log_message(&format!("Creating new simulation with radius={}, height={}, T={}K", cylinder_radius, cylinder_height, temperature));
            
            if let Some(core) = compute_core {
                let sim = VortexSimulation::new_with_compute_core(
                    cylinder_radius,
                    cylinder_height,
                    temperature,
                    core);
                sim
            } else {
                VortexSimulation::new(cylinder_radius, cylinder_height, temperature)
            }
        };
        
        // Process external field parameters and apply if specified
        // (only for new simulations or if we want to override the checkpoint's field)
        if !matches.get_one::<String>("ext_field").unwrap().eq_ignore_ascii_case("none") {
            let ext_field_type = matches.get_one::<String>("ext_field").unwrap().as_str();
            let field_value = matches.get_one::<String>("field_value").unwrap().as_str();
            let field_center = matches.get_one::<String>("field_center").unwrap().as_str();
            let frequency = *matches.get_one::<f64>("frequency").unwrap();
            let phase = *matches.get_one::<f64>("phase").unwrap();
            
            simulation::log_message(&format!("Setting external field: {}", ext_field_type));
            if ext_field_type.eq_ignore_ascii_case("rotation") {
                simulation::log_message(&format!("  Value: {}", field_value));
            } else if ext_field_type.eq_ignore_ascii_case("oscillatory") {
                simulation::log_message(&format!("  Value: {}, Frequency: {} Hz, Phase: {} rad", field_value, frequency, phase));
            }
            
            match simulation::parse_external_field(
                ext_field_type,
                field_value,
                field_center,
                frequency,
                phase
            ) {
                Ok(ext_field) => {
                    sim.external_field = ext_field;
                },
                Err(e) => {
                    simulation::log_message(&format!("Failed to parse external field: {}", e));
                    return Err(e);
                }
            }
        }
        
        // Add Kelvin waves if requested
        let wave_amplitude = matches.get_one::<String>("waves").unwrap().parse::<f64>()?;
        if wave_amplitude > 0.0 {
            simulation::log_message(&format!("Adding Kelvin waves with amplitude = {}", wave_amplitude));
            sim.add_kelvin_waves(wave_amplitude, 3.0);
        }

        // After your simulation is created
        if let Some(interval_str) = matches.get_one::<String>("time_series") {
            if let Ok(interval) = interval_str.parse::<usize>() {
                let series_name = matches.get_one::<String>("time_series_name").unwrap();
                simulation::log_message(&format!("Enabling time series output every {} steps with base name '{}'", interval, series_name));
                sim.save_time_series_vtk(series_name, interval);
            }
        }
        
        // Run the simulation
        sim.run(steps);
        
        // Save the results
        sim.save_results(output);
        
        println!("Results saved to {}", output);
    }
    // Handle resume command in single simulation mode
    else if let Some(matches) = matches.subcommand_matches("resume") {
        let checkpoint_file = matches.get_one::<String>("checkpoint").unwrap();
        let use_gpu = matches.get_flag("gpu");
        let selected_gpu = matches.get_one::<String>("select_gpu").map(|s| s.as_str());
        let steps = matches.get_one::<String>("steps").unwrap().parse::<usize>()?;
        let output = matches.get_one::<String>("output").unwrap();
        
        simulation::log_message(&format!("Resuming simulation from checkpoint: {}", checkpoint_file));
        
        // Load the checkpoint
        let mut sim = match VortexSimulation::load_checkpoint(checkpoint_file) {
            Ok(sim) => sim,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {}", e);
                return Err(Box::new(e));
            }
        };
        
        // Initialize GPU compute core if needed
        if use_gpu {
            simulation::log_message(&format!("Initializing GPU compute core..."));
            let core = pollster::block_on(ComputeCore::new_with_device_preference(selected_gpu));
            sim.set_compute_core(core);
        }
        
        // Continue the simulation
        simulation::log_message(&format!("Continuing simulation for {} more steps...", steps));
        sim.run(steps);
        
        // Save the results
        sim.save_results(output);
        
        println!("\nResumed results saved to {}", output);
    }
    // Handle parameter study
    else if let Some(matches) = matches.subcommand_matches("study") {
        // If list-gpus flag is set, list available GPUs and exit
        if matches.get_flag("list_gpus") {
            println!("Available GPUs:");
            let gpu_list = pollster::block_on(ComputeCore::list_available_gpus());
            for (i, (name, device_type, compatibility)) in gpu_list.iter().enumerate() {
                println!("  {}. {} ({}) - {}", i+1, name, device_type_to_string(*device_type), compatibility);
            }
            return Ok(());
        }

        let radius_min = matches.get_one::<String>("radius_min").unwrap().parse::<f64>()?;
        let radius_max = matches.get_one::<String>("radius_max").unwrap().parse::<f64>()?;
        let radius_steps = matches.get_one::<String>("radius_steps").unwrap().parse::<usize>()?;
        
        let temp_min = matches.get_one::<String>("temperature_min").unwrap().parse::<f64>()?;
        let temp_max = matches.get_one::<String>("temperature_max").unwrap().parse::<f64>()?;
        let temp_steps = matches.get_one::<String>("temperature_steps").unwrap().parse::<usize>()?;
        
        let steps = matches.get_one::<String>("steps").unwrap().parse::<usize>()?;
        let output_dir = matches.get_one::<String>("output_dir").unwrap();
        let wave_amplitude = matches.get_one::<String>("waves").unwrap().parse::<f64>()?;
        
        let use_gpu = matches.get_flag("gpu");
        let selected_gpu = matches.get_one::<String>("select_gpu").map(|s| s.as_str());
        
        // Create output directory if it doesn't exist
        create_dir_all(output_dir)?;
        
        // Initialize GPU compute core if needed
        let compute_core = if use_gpu {
            println!("Initializing GPU compute core...");
            let core = pollster::block_on(ComputeCore::new_with_device_preference(selected_gpu));
            Some(core)
        } else {
            None
        };
        
        // Parse external field if specified
        let ext_field_type = matches.get_one::<String>("ext_field").unwrap().as_str();
        let ext_field = if !ext_field_type.eq_ignore_ascii_case("none") {
            let field_value = matches.get_one::<String>("field_value").unwrap().as_str();
            
            match simulation::parse_external_field(
                ext_field_type,
                field_value,
                "0,0,0", // Default center
                1.0,     // Default frequency
                0.0      // Default phase
            ) {
                Ok(field) => field,
                Err(e) => {
                    eprintln!("Failed to parse external field: {}", e);
                    return Err(e);
                }
            }
        } else {
            None
        };
        
        println!("Running parameter study: radius [{} to {}, {} steps], temp [{} to {}, {} steps]",
                 radius_min, radius_max, radius_steps, temp_min, temp_max, temp_steps);
        
        // Run the parameter study
        let results = simulation::run_parameter_study(
            1.0, // Base radius
            2.0, // Height
            1.5, // Base temperature
            (radius_min, radius_max, radius_steps),
            (temp_min, temp_max, temp_steps),
            wave_amplitude,
            steps,
            output_dir,
            compute_core,
            ext_field,
        );
        
        // Save results summary
        let summary_file = Path::new(output_dir).join("summary.json");
        let file = std::fs::File::create(&summary_file)?;
        serde_json::to_writer_pretty(file, &results)?;
        
        println!("Parameter study complete! Summary saved to {}", summary_file.display());
    }
    // Default behavior if no subcommand is provided
    else {
        eprintln!("No command specified. Use --help for usage information.");
        return Err("No command specified".into());
    }

    Ok(())
}

// Helper function to format device type
fn device_type_to_string(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => "Discrete GPU",
        wgpu::DeviceType::IntegratedGpu => "Integrated GPU",
        wgpu::DeviceType::Cpu => "CPU",
        wgpu::DeviceType::Other => "Other",
        _ => "Unknown",
    }
}
