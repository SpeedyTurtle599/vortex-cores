// External crates
use crate::physics;
use crate::extfields::ExternalField;
use nalgebra::Vector3;
use std::vec::Vec;
use std::fs::File;
use std::io::{self, Write};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

// MARK: Data Structures
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
        println!("Running simulation for {} steps...", steps);
        
        // Initialize vortices
        self.initialize_vortices();

        // Add Kelvin waves with amplitude proportional to temperature
        // More thermal energy = more waves
        if self.temperature > 0.1 {
            let amplitude = 0.05 * self.radius * (self.temperature / 2.17);
            let wavenumber = 2.0 + (self.temperature * 2.0); // More waves at higher temp
            self.add_kelvin_waves(amplitude, wavenumber);
        }
        
        // Set up progress bar
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
        
        // Create more varied and interesting initial state
        
        // First ring
        let ring_radius = self.radius * 0.5;
        let center = [0.0, 0.0, self.height * 0.3];
        let mut ring1 = self.create_vortex_ring(ring_radius, center);
        physics::add_kelvin_wave(&mut ring1, 0.15 * ring_radius, 4);
        self.vortex_lines.push(ring1);
        
        // Second ring (titled)
        let ring2_radius = self.radius * 0.45;
        let center2 = [0.0, 0.0, self.height * 0.7];
        let mut ring2 = self.create_vortex_ring(ring2_radius, center2);
        
        // Tilt the ring
        for point in &mut ring2.points {
            point.position[0] += 0.1 * self.radius;
            point.position[2] -= 0.1 * (point.position[0] / ring2_radius) * self.radius;
        }
        
        physics::add_kelvin_wave(&mut ring2, 0.1 * ring2_radius, 3);
        physics::update_tangent_vectors(&mut ring2);
        self.vortex_lines.push(ring2);
        
        // Add a helical vortex line
        let helix = self.create_helix(
            self.radius * 0.3,
            80,
            4.0 * std::f64::consts::PI,
            0.1 * self.height,
            0.9 * self.height
        );
        self.vortex_lines.push(helix);
        
        // Add a vortex loop with figure-8 shape
        let mut figure8 = self.create_figure8(
            self.radius * 0.4,
            self.radius * 0.25,
            [0.0, 0.0, self.height * 0.5],
            60
        );
        physics::update_tangent_vectors(&mut figure8);
        self.vortex_lines.push(figure8);
        
        // If temperature is high, add some random vortex filaments
        if self.temperature > 1.5 {
            self.add_random_vortex_filaments(3 + (self.temperature as usize));
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
        
        VortexLine{points}
    }

    // Helper method to create a helix
    fn create_helix(&self, radius: f64, num_points: usize, total_angle: f64, 
                    z_start: f64, z_end: f64) -> VortexLine {
        let mut points = Vec::with_capacity(num_points);
        
        for i in 0..num_points {
            let t = i as f64 / (num_points - 1) as f64;
            let angle = t * total_angle;
            let z = z_start + t * (z_end - z_start);
            
            points.push(VortexPoint {
                position: [
                    radius * angle.cos(),
                    radius * angle.sin(),
                    z
                ],
                tangent: [0.0, 0.0, 0.0], // Will be calculated later
            });
        }
        
        let mut helix = VortexLine{points};
        physics::update_tangent_vectors(&mut helix);
        helix
    }
    
    // Helper method to create a figure-8 shape
    fn create_figure8(&self, radius1: f64, radius2: f64, center: [f64; 3], num_points: usize) -> VortexLine {
        let mut points = Vec::with_capacity(num_points);
        
        for i in 0..num_points {
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            
            // Figure-8 parametric equation
            let x = center[0] + radius1 * t.sin();
            let y = center[1] + radius2 * t.sin() * t.cos();
            let z = center[2] + radius2 * t.cos();
            
            points.push(VortexPoint {
                position: [x, y, z],
                tangent: [0.0, 0.0, 0.0], // Will be calculated later
            });
        }
        
        VortexLine{points}
    }
    
    // Add random vortex filaments
    fn add_random_vortex_filaments(&mut self, count: usize) {
        let mut rng = rand::rng();
        
        for _ in 0..count {
            // Create a random curved line
            let num_points = 30 + (rng.random::<f64>() * 30.0) as usize;
            let mut points = Vec::with_capacity(num_points);
            
            // Start at a random position
            let start_x = (rng.random::<f64>() * 2.0 - 1.0) * self.radius * 0.8;
            let start_y = (rng.random::<f64>() * 2.0 - 1.0) * self.radius * 0.8;
            let start_z = (rng.random::<f64>() * self.height) * 0.8 + 0.1 * self.height;
            
            // Random direction
            let dir_x = rng.random::<f64>() * 2.0 - 1.0;
            let dir_y = rng.random::<f64>() * 2.0 - 1.0;
            let dir_z = rng.random::<f64>() * 2.0 - 1.0;
            let dir_mag = (dir_x*dir_x + dir_y*dir_y + dir_z*dir_z).sqrt();
            
            let dir_x = dir_x / dir_mag;
            let dir_y = dir_y / dir_mag;
            let dir_z = dir_z / dir_mag;
            
            // Create a curved path
            for i in 0..num_points {
                let t = i as f64 / (num_points - 1) as f64;
                
                // Add some randomness to the path
                let wobble_x = (t * 7.0).sin() * 0.1 * self.radius;
                let wobble_y = (t * 8.0).cos() * 0.1 * self.radius;
                let wobble_z = (t * 5.0).sin() * 0.1 * self.radius;
                
                let x = start_x + t * dir_x * self.radius * 1.6 + wobble_x;
                let y = start_y + t * dir_y * self.radius * 1.6 + wobble_y;
                let z = start_z + t * dir_z * self.height * 0.8 + wobble_z;
                
                // Keep within bounds
                let r_squared = x*x + y*y;
                if r_squared > self.radius * self.radius * 0.9 || 
                   z < 0.05 * self.height || z > 0.95 * self.height {
                    continue;
                }
                
                points.push(VortexPoint {
                    position: [x, y, z],
                    tangent: [0.0, 0.0, 0.0], // Will be calculated later
                });
            }
            
            if points.len() >= 3 {
                let mut line = VortexLine { points };
                physics::update_tangent_vectors(&mut line);
                self.vortex_lines.push(line);
            }
        }
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

    fn add_thermal_fluctuations(&mut self, dt: f64) {
        // Only apply thermal fluctuations if temperature is above absolute zero
        if self.temperature < 0.01 {
            return;
        }
        
        // Get Hall-Vinen mutual friction coefficients
        let (alpha, alpha_prime) = physics::mutual_friction_coefficients(self.temperature);
        
        // Temperature-dependent noise amplitude with more physical basis
        let noise_amplitude = 1e-4 * (self.temperature / 2.17).sqrt() * dt.sqrt() * alpha;
        let mut rng = rand::rng();
        
        // First, calculate velocities for all lines ahead of time
        // This avoids the borrowing conflict
        let mut all_velocities = Vec::with_capacity(self.vortex_lines.len());
        
        for line in &self.vortex_lines {
            let velocities = physics::calculate_local_velocities(line, &self.vortex_lines);
            all_velocities.push(velocities);
        }
        
        // Now apply the fluctuations using pre-calculated velocities
        for (line_idx, line) in self.vortex_lines.iter_mut().enumerate() {
            let velocities = &all_velocities[line_idx];
            
            for (i, point) in line.points.iter_mut().enumerate() {
                // Get local velocity components
                let v_sl = velocities.get(i).unwrap_or(&Vector3::zeros()).clone();
                
                // Apply mutual friction
                let tangent = Vector3::new(point.tangent[0], point.tangent[1], point.tangent[2]);
                let v_sl_perp = v_sl - tangent * v_sl.dot(&tangent);
                
                // Apply temperature-dependent friction
                let friction = alpha * v_sl_perp.cross(&tangent) + alpha_prime * v_sl_perp;
                
                // Create random vector with physically correct distribution
                let random_vec = Vector3::new(
                    rng.random::<f64>() * 2.0 - 1.0, // Use gen instead of random
                    rng.random::<f64>() * 2.0 - 1.0,
                    rng.random::<f64>() * 2.0 - 1.0
                );
                
                // Project random vector to be perpendicular to tangent
                let dot = random_vec.dot(&tangent);
                let perpendicular = random_vec - tangent * dot;
                
                // Normalize and scale by noise amplitude
                let noise = perpendicular.normalize() * noise_amplitude;
                
                // Apply both deterministic friction and random fluctuations
                point.position[0] += (friction.x + noise.x) * dt;
                point.position[1] += (friction.y + noise.y) * dt;
                point.position[2] += (friction.z + noise.z) * dt;
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
                        _ => {} // No additional potential for other field types
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
        
        // Use the detailed VTK export instead of basic save_vtk
        if let Err(e) = self.save_detailed_vtk(filename) {
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

pub fn run_parameter_study(
    base_radius: f64,
    base_height: f64,
    base_temperature: f64,
    radius_range: (f64, f64, usize),       // (min, max, steps)
    temperature_range: (f64, f64, usize),  // (min, max, steps)
    wave_amplitude: f64,
    steps: usize,
    output_dir: &str,
) -> Vec<SimulationResult> {
    let mut results = Vec::new();
    
    let radii = generate_parameter_range(radius_range.0, radius_range.1, radius_range.2);
    let temperatures = generate_parameter_range(temperature_range.0, temperature_range.1, temperature_range.2);
    
    let total_runs = radii.len() * temperatures.len();
    println!("Running parameter study with {} total configurations", total_runs);
    
    let progress_bar = ProgressBar::new(total_runs as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    
    for (i, &radius) in radii.iter().enumerate() {
        for (j, &temperature) in temperatures.iter().enumerate() {
            let config_name = format!("R{:.2}_T{:.2}", radius, temperature);
            progress_bar.set_message(format!("Running configuration {}", config_name));
            
            // Create and run simulation with these parameters
            let mut sim = VortexSimulation::new(radius, base_height, temperature);
            if wave_amplitude > 0.0 {
                sim.add_kelvin_waves(wave_amplitude * sim.radius, 3.0);
            }
            sim.run(steps);
            
            // Save results
            let output_file = format!("{}/{}_{}.vtk", output_dir, i, j);
            sim.save_results(&output_file);
            
            // Collect key metrics
            let result = SimulationResult {
                radius,
                temperature, 
                final_length: *sim.stats.total_length.last().unwrap_or(&0.0),
                final_energy: *sim.stats.kinetic_energy.last().unwrap_or(&0.0),
                reconnection_count: sim.stats.reconnection_count,
            };
            
            results.push(result);
            progress_bar.inc(1);
        }
    }
    
    progress_bar.finish();
    results
}

fn generate_parameter_range(min: f64, max: f64, steps: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(steps);
    if steps <= 1 {
        values.push(min);
        return values;
    }
    
    let step_size = (max - min) / ((steps - 1) as f64);
    for i in 0..steps {
        values.push(min + (i as f64) * step_size);
    }
    values
}

// MARK: Simulation Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub radius: f64,
    pub temperature: f64,
    pub final_length: f64,
    pub final_energy: f64,
    pub reconnection_count: usize,
}