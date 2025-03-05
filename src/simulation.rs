use crate::physics;
use crate::visualisation;
use crate::extfields::ExternalField;
use nalgebra::Vector3;
use std::vec::Vec;
use std::fs::File;
use std::io::{self, Write};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexPoint {
    pub position: [f64; 3],
    pub tangent: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexLine {
    pub points: Vec<VortexPoint>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VortexSimulation {
    pub radius: f64,
    pub height: f64,
    pub temperature: f64,
    pub vortex_lines: Vec<VortexLine>,
    pub time: f64,
    pub external_field: Option<ExternalFieldParams>,
    pub stats: SimulationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternalFieldParams {
    Rotation {
        angular_velocity: [f64; 3], // rad/s
        center: [f64; 3],           // cm
    },
    UniformFlow {
        velocity: [f64; 3],         // cm/s
    },
    OscillatoryFlow {
        amplitude: [f64; 3],        // cm/s
        frequency: f64,             // Hz
        phase: f64,                 // rad
    },
    Counterflow {
        velocity: [f64; 3],         // cm/s
    },
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SimulationStats {
    pub total_length: Vec<f64>,
    pub kinetic_energy: Vec<f64>,
    pub reconnection_count: usize,
    pub time_points: Vec<f64>,
}

// MARK: Implementation
impl VortexSimulation {
    pub fn new(radius: f64, height: f64, temperature: f64) -> Self {
        VortexSimulation {
            radius,
            height,
            temperature,
            vortex_lines: Vec::new(),
            time: 0.0,
            external_field: None,
            stats: SimulationStats::default(),
        }
    }
    
    pub fn set_external_field(&mut self, field: ExternalField) {
        // Convert ExternalField to storable ExternalFieldParams
        self.external_field = match field {
            ExternalField::Rotation { angular_velocity, center } => Some(ExternalFieldParams::Rotation {
                angular_velocity: [angular_velocity.x, angular_velocity.y, angular_velocity.z],
                center: [center.x, center.y, center.z],
            }),
            ExternalField::UniformFlow { velocity } => Some(ExternalFieldParams::UniformFlow {
                velocity: [velocity.x, velocity.y, velocity.z],
            }),
            ExternalField::OscillatoryFlow { amplitude, frequency, phase } => Some(ExternalFieldParams::OscillatoryFlow {
                amplitude: [amplitude.x, amplitude.y, amplitude.z],
                frequency,
                phase,
            }),
            ExternalField::Counterflow { velocity } => Some(ExternalFieldParams::Counterflow {
                velocity: [velocity.x, velocity.y, velocity.z],
            }),
            _ => None, // Custom field cannot be stored directly
        };
    }
    
    fn get_external_field(&self) -> Option<ExternalField> {
        match &self.external_field {
            Some(ExternalFieldParams::Rotation { angular_velocity, center }) => {
                Some(ExternalField::Rotation {
                    angular_velocity: Vector3::new(angular_velocity[0], angular_velocity[1], angular_velocity[2]),
                    center: Vector3::new(center[0], center[1], center[2]),
                })
            },
            Some(ExternalFieldParams::UniformFlow { velocity }) => {
                Some(ExternalField::UniformFlow {
                    velocity: Vector3::new(velocity[0], velocity[1], velocity[2]),
                })
            },
            Some(ExternalFieldParams::OscillatoryFlow { amplitude, frequency, phase }) => {
                Some(ExternalField::OscillatoryFlow {
                    amplitude: Vector3::new(amplitude[0], amplitude[1], amplitude[2]),
                    frequency: *frequency,
                    phase: *phase,
                })
            },
            Some(ExternalFieldParams::Counterflow { velocity }) => {
                Some(ExternalField::Counterflow {
                    velocity: Vector3::new(velocity[0], velocity[1], velocity[2]),
                })
            },
            None => None,
        }
    }
    
    pub fn run(&mut self, steps: usize) {
        println!("Running simulation for {steps} steps...");
        
        // Initialize vortices
        self.initialize_vortices();
        
        // Setup progress bar
        let progress_bar = ProgressBar::new(steps as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        
        // Time evolution loop
        let dt = 0.001; // time step
        let remesh_interval = 10; // Remesh every 10 steps
        let checkpoint_interval = 100; // Save checkpoint every 100 steps
        
        for step in 0..steps {
            progress_bar.set_position(step as u64);
            
            if step % 10 == 0 {
                progress_bar.set_message(format!("Total length: {:.4} cm", 
                    self.stats.total_length.last().unwrap_or(&0.0)));
            }
            
            // Calculate vortex dynamics
            self.evolve_vortices(dt);
            
            // Handle vortex reconnections
            self.handle_reconnections();
            
            // Apply boundary conditions
            self.apply_boundary_conditions();
            
            // Periodically remesh points to maintain stability
            if step % remesh_interval == 0 {
                self.remesh_vortices();
            }
            
            // Update statistics
            if step % 10 == 0 {
                self.update_statistics();
            }
            
            // Save checkpoint
            if step % checkpoint_interval == 0 && step > 0 {
                let checkpoint_filename = format!("checkpoint_{}.json", step);
                if let Err(e) = self.save_checkpoint(&checkpoint_filename) {
                    eprintln!("Error saving checkpoint: {}", e);
                }
            }
            
            self.time += dt;
        }
        
        progress_bar.finish_with_message("Simulation complete!");
    }
    
    fn initialize_vortices(&mut self) {
        println!("Initializing vortex configuration...");
        
        // Create a simple vortex ring
        let ring_radius = self.radius * 0.5;
        let center = [0.0, 0.0, self.height * 0.5];
        let mut ring = self.create_vortex_ring(ring_radius, center);
        
        // Add some Kelvin waves if needed
        if let Some(params) = &self.external_field {
            match params {
                ExternalFieldParams::Rotation { .. } => {
                    // For rotating superfluid, create vortex array
                    self.create_vortex_array();
                    return;
                },
                _ => {
                    // Add some Kelvin waves to make it more interesting
                    physics::add_kelvin_wave(&mut ring, 0.1, 5);
                }
            }
        }
        
        // Add more complex configurations depending on parameters
        self.vortex_lines.push(ring);
        
        // Add a second ring to promote reconnections and tangle formation
        let ring2_radius = self.radius * 0.4;
        let center2 = [0.0, 0.0, self.height * 0.7];
        let ring2 = self.create_vortex_ring(ring2_radius, center2);
        self.vortex_lines.push(ring2);
    }
    
    fn create_vortex_array(&mut self) {
        // Create array of vortices for rotation
        let num_vortices = 10; // Arbitrary number for demonstration
        let mut rng = rand::rng();
        
        for _ in 0..num_vortices {
            // Place vortices randomly within the cylinder
            let r = self.radius * 0.8 * rng.random::<f64>().sqrt(); // sqrt for uniform distribution in circle
            let theta = 2.0 * std::f64::consts::PI * rng.random::<f64>();
            let x = r * theta.cos();
            let y = r * theta.sin();
            let z_start = 0.1 * self.height;
            let z_end = 0.9 * self.height;
            
            // Create a straight vortex line
            let mut points = Vec::new();
            let num_points = 50;
            
            for i in 0..num_points {
                let z = z_start + (z_end - z_start) * (i as f64) / ((num_points - 1) as f64);
                
                points.push(VortexPoint {
                    position: [x, y, z],
                    tangent: [0.0, 0.0, 1.0],
                });
            }
            
            self.vortex_lines.push(VortexLine { points });
        }
    }
    
    fn create_vortex_ring(&self, ring_radius: f64, center: [f64; 3]) -> VortexLine {
        let num_points = 100;
        let mut points = Vec::with_capacity(num_points);
        
        for i in 0..num_points {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            
            let x = center[0] + ring_radius * theta.cos();
            let y = center[1] + ring_radius * theta.sin();
            let z = center[2];
            
            // Tangent is perpendicular to radius and in the x-y plane
            let tx = -theta.sin();
            let ty = theta.cos();
            let tz = 0.0;
            
            points.push(VortexPoint {
                position: [x, y, z],
                tangent: [tx, ty, tz],
            });
        }
        
        VortexLine { points }
    }
    
    fn evolve_vortices(&mut self, dt: f64) {
        // Get external field for this time step
        let ext_field = self.get_external_field();
        
        // Add thermal fluctuations if temperature > 0
        if self.temperature > 0.01 {
            self.add_thermal_fluctuations(dt);
        }
        
        // Evolve the vortex network with parallel computation
        physics::evolve_vortex_network(
            &mut self.vortex_lines, 
            dt, 
            self.temperature,
            ext_field.as_ref(),
            self.time
        );
    }

    // Add this new method to implement thermal fluctuations
    fn add_thermal_fluctuations(&mut self, dt: f64) {
        // Only apply thermal fluctuations if temperature is above absolute zero
        if self.temperature < 0.01 {
            return;
        }
        
        // Temperature-dependent noise amplitude
        // This is a simplified model based on fluctuation-dissipation theorem
        // Real model would be more complex
        let noise_amplitude = 1e-4 * (self.temperature / 2.17).sqrt() * dt.sqrt();
        let mut rng = rand::rng();
        
        for line in &mut self.vortex_lines {
            for point in &mut line.points {
                // Generate random fluctuations perpendicular to vortex line
                // (to preserve line length approximately)
                
                // Create random vector
                let random_vec = Vector3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0), 
                    rng.gen_range(-1.0..1.0)
                );
                
                // Get tangent as Vector3
                let tangent = Vector3::new(
                    point.tangent[0],
                    point.tangent[1],
                    point.tangent[2]
                );
                
                // Project random vector to be perpendicular to tangent
                let dot = random_vec.dot(&tangent);
                let perpendicular = random_vec - tangent * dot;
                
                // Normalize and scale by noise amplitude
                let noise = perpendicular.normalize() * noise_amplitude;
                
                // Apply perturbation
                point.position[0] += noise.x;
                point.position[1] += noise.y;
                point.position[2] += noise.z;
            }
        }
        
        // Update tangent vectors after applying noise
        for line in &mut self.vortex_lines {
            physics::update_tangent_vectors(line);
        }
    }
    
    fn remesh_vortices(&mut self) {
        let target_spacing = self.radius / 50.0;
        let min_spacing = target_spacing * 0.5;
        let max_spacing = target_spacing * 2.0;
        
        for line in &mut self.vortex_lines {
            physics::remesh_vortex_line(line, target_spacing, min_spacing, max_spacing);
        }
    }
    
    fn handle_reconnections(&mut self) {
        let reconnection_threshold = 0.01 * self.radius;
        let previous_line_count = self.vortex_lines.len();
        let previous_total_length = physics::calculate_total_length(&self.vortex_lines);
        
        // The function may not return reconnection count, so we need to check before and after
        physics::handle_reconnections(&mut self.vortex_lines, reconnection_threshold);
        
        // After reconnection, check if line count or length changed significantly
        let current_line_count = self.vortex_lines.len();
        let current_total_length = physics::calculate_total_length(&self.vortex_lines);
        
        let line_count_changed = current_line_count != previous_line_count;
        let length_reduced = (previous_total_length - current_total_length) > 0.001;
        
        if line_count_changed || length_reduced {
            // A reconnection likely happened
            self.stats.reconnection_count += 1;
            println!(
                "Reconnection detected at time={:.4}: Lines: {} → {}, Length: {:.4} → {:.4}",
                self.time,
                previous_line_count,
                current_line_count,
                previous_total_length,
                current_total_length
            );
        }
    }
    
    fn apply_boundary_conditions(&mut self) {
        // Apply cylindrical boundary conditions
        for line in &mut self.vortex_lines {
            for point in &mut line.points {
                // Check cylinder walls (radial boundary)
                let x = point.position[0];
                let y = point.position[1];
                let r_squared = x*x + y*y;
                
                if r_squared > self.radius * self.radius {
                    // Reflect back into cylinder
                    let r = r_squared.sqrt();
                    let factor = 2.0 * self.radius / r - 1.0;
                    point.position[0] = x * factor;
                    point.position[1] = y * factor;
                }
                
                // Check top and bottom boundaries
                if point.position[2] < 0.0 {
                    point.position[2] = -point.position[2];
                } else if point.position[2] > self.height {
                    point.position[2] = 2.0 * self.height - point.position[2];
                }
            }
        }
    }
    
    fn calculate_detailed_energy(&self) -> (f64, f64, f64) {
        // Constants for superfluid helium
        let rho_s = 0.145; // g/cm³, superfluid density at ~1.5K
        let kappa = 9.97e-4; // quantum of circulation, cm²/s
        
        // Calculate kinetic energy from the vortex configuration
        let e_kinetic = physics::calculate_kinetic_energy(&self.vortex_lines);
        
        // Calculate potential energy from external field
        let mut e_potential = 0.0;
        
        if let Some(ext_field) = self.get_external_field() {
            // Calculate for each vortex segment
            for line in &self.vortex_lines {
                for point in &line.points {
                    // Convert position to Vector3
                    let pos = Vector3::new(
                        point.position[0], 
                        point.position[1], 
                        point.position[2]
                    );
                    
                    // Get external velocity at this point
                    let v_ext = ext_field.velocity_at(&pos, self.time);
                    
                    // Contribution to potential energy depends on field type
                    match &ext_field {
                        ExternalField::Rotation { angular_velocity, center } => {
                            let r_vec = pos - center;
                            let r_perp = Vector3::new(r_vec.x, r_vec.y, 0.0); // perpendicular component
                            let omega = angular_velocity.norm();
                            
                            // Potential energy in rotation frame (simplified model)
                            e_potential += -rho_s * kappa * omega * r_perp.norm_squared() / 2.0;
                        },
                        // Add other field types as needed
                        _ => {}
                    }
                }
            }
        }
        
        // Calculate wave energy (from Kelvin waves)
        let e_waves = physics::calculate_kelvin_wave_energy(&self.vortex_lines);
        
        (e_kinetic, e_potential, e_waves)
    }
    
    fn update_statistics(&mut self) {
        // Calculate total vortex length
        let total_length = physics::calculate_total_length(&self.vortex_lines);
        
        // Calculate kinetic energy
        let kinetic_energy = physics::calculate_kinetic_energy(&self.vortex_lines);
        
        // Store the values
        self.stats.total_length.push(total_length);
        self.stats.kinetic_energy.push(kinetic_energy);
        self.stats.time_points.push(self.time);
        
        // Every 10 updates, calculate and print detailed energy breakdown
        if self.stats.time_points.len() % 10 == 0 {
            let (e_kin, e_pot, e_wave) = self.calculate_detailed_energy();
            println!("Energy: Kinetic={:.4e}, Potential={:.4e}, Waves={:.4e}", e_kin, e_pot, e_wave);
        }
    }
    
    // Save checkpoint for resuming simulation later
    pub fn save_checkpoint(&self, filename: &str) -> io::Result<()> {
        println!("Saving checkpoint to {}...", filename);
        let file = File::create(filename)?;
        serde_json::to_writer(file, self)?;
        Ok(())
    }
    
    // Load checkpoint to resume simulation
    pub fn load_checkpoint(filename: &str) -> io::Result<Self> {
        println!("Loading checkpoint from {}...", filename);
        let file = File::open(filename)?;
        let sim: VortexSimulation = serde_json::from_reader(file)?;
        Ok(sim)
    }
    
    // Save simulation results
    pub fn save_results(&self, filename: &str) {
        println!("Saving results to {}...", filename);
        
        // Save VTK file for visualization
        if let Err(e) = visualisation::save_vtk(self, filename) {
            eprintln!("Error saving VTK file: {}", e);
        }
        
        // Save statistics to JSON
        let stats_filename = filename.replace(".vtk", "_stats.json");
        if let Err(e) = self.save_statistics(&stats_filename) {
            eprintln!("Error saving statistics: {}", e);
        }
    }
    
    // Save detailed statistics
    fn save_statistics(&self, filename: &str) -> io::Result<()> {
        let file = File::create(filename)?;
        serde_json::to_writer_pretty(file, &self.stats)?;
        Ok(())
    }
    
    // Function to add Kelvin waves to an existing vortex configuration
    pub fn add_kelvin_waves(&mut self, amplitude: f64, wavenumber: f64) {
        for line in &mut self.vortex_lines {
            // Determine primary axis of the line
            let n_points = line.points.len();
            if n_points < 3 {
                continue;
            }
            
            // Get approximate direction from first and middle point
            let mid_idx = n_points / 2;
            let start = Vector3::new(
                line.points[0].position[0],
                line.points[0].position[1],
                line.points[0].position[2]
            );
            
            let mid = Vector3::new(
                line.points[mid_idx].position[0],
                line.points[mid_idx].position[1],
                line.points[mid_idx].position[2]
            );
            
            let direction = (mid - start).normalize();
            
            // Convert wavenumber (which might be fractional) to wavelengths (integer)
            let wavelengths = (wavenumber * n_points as f64 / (2.0 * std::f64::consts::PI)).round() as usize;
            
            // Add Kelvin wave
            physics::add_kelvin_wave(
                line,
                amplitude,
                wavelengths // Now correctly passing a usize
            );
        }
    }
    
    // Generate a VTK file visualizing the vortex tangle with additional attributes
    pub fn save_detailed_vtk(&self, filename: &str) -> io::Result<()> {
        // This is a more advanced VTK export that includes various scalar fields
        // like local curvature, velocity, etc.
        
        let mut file = File::create(filename)?;
        
        // VTK header
        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "Superfluid Helium Vortex Tangle Simulation")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        
        // Count total points
        let mut total_points = 0;
        for line in &self.vortex_lines {
            total_points += line.points.len();
        }
        
        // Write points
        writeln!(file, "POINTS {} float", total_points)?;
        for line in &self.vortex_lines {
            for point in &line.points {
                writeln!(file, "{} {} {}", point.position[0], point.position[1], point.position[2])?;
            }
        }
        
        // Write cells (lines)
        let mut total_cells = 0;
        let mut total_list_size = 0;
        for line in &self.vortex_lines {
            let n_points = line.points.len();
            total_cells += n_points - 1;  // segments between points
            total_list_size += (n_points - 1) * 3;  // 3 values per segment (type + 2 points)
        }
        
        writeln!(file, "CELLS {} {}", total_cells, total_list_size)?;
        
        let mut point_offset = 0;
        for line in &self.vortex_lines {
            let n_points = line.points.len();
            for i in 0..n_points-1 {
                writeln!(file, "2 {} {}", point_offset + i, point_offset + i + 1)?;
            }
            point_offset += n_points;
        }
        
        // Write cell types (type 3 = line)
        writeln!(file, "CELL_TYPES {}", total_cells)?;
        for _ in 0..total_cells {
            writeln!(file, "3")?;
        }
        
        // Write point data
        writeln!(file, "POINT_DATA {}", total_points)?;
        
        // Tangent vectors
        writeln!(file, "VECTORS tangent float")?;
        for line in &self.vortex_lines {
            for point in &line.points {
                writeln!(file, "{} {} {}", point.tangent[0], point.tangent[1], point.tangent[2])?;
            }
        }
        
        // Calculate and write local curvature
        writeln!(file, "SCALARS curvature float 1")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        
        for line in &self.vortex_lines {
            let n_points = line.points.len();
            for i in 0..n_points {
                let prev = (i + n_points - 1) % n_points;
                let next = (i + 1) % n_points;
                
                let prev_pos = Vector3::new(
                    line.points[prev].position[0],
                    line.points[prev].position[1],
                    line.points[prev].position[2]
                );
                
                let pos = Vector3::new(
                    line.points[i].position[0],
                    line.points[i].position[1],
                    line.points[i].position[2]
                );
                
                let next_pos = Vector3::new(
                    line.points[next].position[0],
                    line.points[next].position[1],
                    line.points[next].position[2]
                );
                
                // Calculate segments
                let segment1 = pos - prev_pos;
                let segment2 = next_pos - pos;
                
                let t1 = segment1.normalize();
                let t2 = segment2.normalize();
                
                // Calculate binormal (t1 × t2)
                let binormal = t1.cross(&t2);
                let bin_mag = binormal.norm();
                
                // Calculate curvature approximation
                let dot = t1.dot(&t2);
                let curvature = bin_mag / (1.0 + dot);
                
                writeln!(file, "{}", curvature)?;
            }
        }
        
        // Add velocity magnitude data
        writeln!(file, "SCALARS velocity_mag float 1")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        
        // Get external field for velocity calculation
        let ext_field = self.get_external_field();
        
        for line in &self.vortex_lines {
            for point in &line.points {
                // Convert position to Vector3
                let pos = Vector3::new(
                    point.position[0],
                    point.position[1],
                    point.position[2]
                );
                
                // Calculate LIA velocity (simplified estimate)
                let curvature = 0.1; // This should be calculated properly
                let lia_vel = 9.97e-4 * 10.0 * curvature; // κ * β * curvature
                
                // Add external velocity if any
                let mut total_vel = lia_vel;
                
                if let Some(field) = &ext_field {
                    let ext_vel = field.velocity_at(&pos, self.time);
                    total_vel += ext_vel.norm();
                }
                
                writeln!(file, "{}", total_vel)?;
            }
        }
        
        // Add temperature data if relevant
        if self.temperature > 0.01 {
            writeln!(file, "SCALARS temperature float 1")?;
            writeln!(file, "LOOKUP_TABLE default")?;
            
            // In a real simulation, temperature might vary in space
            // Here we use a constant value for simplicity
            for _ in 0..total_points {
                writeln!(file, "{}", self.temperature)?;
            }
        }
        
        Ok(())
    }
}