// External crates
use crate::physics;
use crate::extfields::ExternalField;
use crate::compute::ComputeCore;
use nalgebra::Vector3;
use std::vec::Vec;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use once_cell::sync::Lazy;

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
    compute_core: Option<ComputeCore>,
    using_gpu: bool,
    pub radius: f64,
    pub height: f64,
    pub temperature: f64,
    pub vortex_lines: Vec<VortexLine>,
    pub time: f64,
    pub external_field: Option<ExternalFieldParams>,
    pub stats: SimulationStats,
    pub time_series_config: Option<TimeSeriesConfig>,
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
// Add this after the other structs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfig {
    pub base_filename: String,
    pub interval: usize,
    pub frame_count: usize,
}

// Global progress bar for coordinated output
static PROGRESS_BAR: Lazy<Arc<Mutex<Option<(MultiProgress, ProgressBar)>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(None))
});

// Add status bars for different metrics
static STATUS_LENGTH: Lazy<Arc<Mutex<Option<ProgressBar>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(None))
});

static STATUS_ENERGY: Lazy<Arc<Mutex<Option<ProgressBar>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(None))
});

static STATUS_CHECKPOINT: Lazy<Arc<Mutex<Option<ProgressBar>>>> = Lazy::new(|| {
    Arc::new(Mutex::new(None))
});

pub fn set_global_progress_bar(multi_progress: MultiProgress, bar: ProgressBar) {
    // First, clear any existing progress bars
    clear_global_progress_bar();
    
    // Now set up the new progress bar
    let mut global_bar = PROGRESS_BAR.lock().unwrap();
    *global_bar = Some((multi_progress.clone(), bar));
    
    // Create and add status bars
    let length_bar = multi_progress.add(ProgressBar::new(1));
    length_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {msg}")
        .unwrap());
    length_bar.set_message("Total length: initializing...");
    
    let energy_bar = multi_progress.add(ProgressBar::new(1));
    energy_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {msg}")
        .unwrap());
    energy_bar.set_message("Energy values: calculating...");
    
    let checkpoint_bar = multi_progress.add(ProgressBar::new(1));
    checkpoint_bar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {msg}")
        .unwrap());
    checkpoint_bar.set_message("No checkpoints saved yet");
    
    // Store status bars
    {
        let mut length = STATUS_LENGTH.lock().unwrap();
        *length = Some(length_bar);
    }
    
    {
        let mut energy = STATUS_ENERGY.lock().unwrap();
        *energy = Some(energy_bar);
    }
    
    {
        let mut checkpoint = STATUS_CHECKPOINT.lock().unwrap();
        *checkpoint = Some(checkpoint_bar);
    }
}

pub fn clear_global_progress_bar() {
    // Clear all status bars first
    {
        let mut length_bar = STATUS_LENGTH.lock().unwrap();
        if let Some(bar) = length_bar.take() {
            bar.finish_and_clear();
        }
    }
    
    {
        let mut energy_bar = STATUS_ENERGY.lock().unwrap();
        if let Some(bar) = energy_bar.take() {
            bar.finish_and_clear();
        }
    }
    
    {
        let mut checkpoint_bar = STATUS_CHECKPOINT.lock().unwrap();
        if let Some(bar) = checkpoint_bar.take() {
            bar.finish_and_clear();
        }
    }
    
    // Clear main progress bar
    let mut global_bar = PROGRESS_BAR.lock().unwrap();
    if let Some((multi_progress, bar)) = global_bar.take() {
        bar.finish();
        drop(multi_progress)
    }
}

pub fn log_message(message: &str) {
    let global_bar = PROGRESS_BAR.lock().unwrap();
    if let Some((_, bar)) = &*global_bar {
        bar.set_message(message.to_string());
    } else {
        println!("{}", message);
    }
}

// New helper functions for specific status updates
pub fn update_length_status(message: &str) {
    let length_bar = STATUS_LENGTH.lock().unwrap();
    if let Some(bar) = &*length_bar {
        bar.set_message(message.to_string());
    } else {
        println!("Length: {}", message);
    }
}

pub fn update_energy_status(message: &str) {
    let energy_bar = STATUS_ENERGY.lock().unwrap();
    if let Some(bar) = &*energy_bar {
        bar.set_message(message.to_string());
    } else {
        println!("Energy: {}", message);
    }
}

pub fn update_checkpoint_status(message: &str) {
    let checkpoint_bar = STATUS_CHECKPOINT.lock().unwrap();
    if let Some(bar) = &*checkpoint_bar {
        bar.set_message(message.to_string());
    } else {
        println!("Checkpoint: {}", message);
    }
}

// MARK: Implementation
impl VortexSimulation {
    pub fn new(radius: f64, height: f64, temperature: f64) -> Self {
        VortexSimulation {
            compute_core: None,
            using_gpu: false,
            radius,
            height,
            temperature,
            vortex_lines: Vec::new(),
            time: 0.0,
            external_field: None,
            stats: SimulationStats::default(),
            time_series_config: None,
        }
    }

    pub async fn new_with_gpu(radius: f64, height: f64, temperature: f64) -> Self {
        let mut sim = Self::new(radius, height, temperature);
        sim.compute_core = Some(ComputeCore::new().await);
        sim.using_gpu = true; // Set to true when GPU is initialized
        sim
    }

    pub fn new_with_compute_core(radius: f64, height: f64, temperature: f64, compute_core: ComputeCore) -> Self {
        let mut sim = Self::new(radius, height, temperature);
        sim.compute_core = Some(compute_core);
        sim.using_gpu = true; // Set to true when GPU is initialized
        sim
    }

    pub fn set_compute_core(&mut self, compute_core: ComputeCore) {
        self.compute_core = Some(compute_core);
        self.using_gpu = true; // Set to true when GPU is initialized
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
        let multi_progress = MultiProgress::new();
        let progress_bar = multi_progress.add(ProgressBar::new(steps as u64));
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        set_global_progress_bar(multi_progress, progress_bar.clone());
        
        progress_bar.set_message("Initializing simulation...");
        
        // Initialize vortices
        self.initialize_vortices();
    
        // Add Kelvin waves with amplitude proportional to temperature
        // More thermal energy = more waves
        if self.temperature > 0.1 {
            let amplitude = 0.05 * self.radius * (self.temperature / 2.17);
            let wavenumber = 2.0 + (self.temperature * 2.0); // More waves at higher temp
            self.add_kelvin_waves(amplitude, wavenumber);
        }
        
        progress_bar.set_message(format!("Running simulation with {} vortex lines", self.vortex_lines.len()));
        
        // Time evolution loop
        let dt = 0.001; // time step
        let remesh_interval = 10; // Remesh every 10 steps
        let checkpoint_interval = 100; // Save checkpoint every 100 steps
        
        for step in 0..steps {
            progress_bar.set_position(step as u64);
            
            // Calculate vortex dynamics
            self.evolve_vortices(dt);
            
            // Handle vortex reconnections
            self.handle_reconnections_with_progress_bar(&progress_bar);
            
            // Apply boundary conditions
            self.apply_boundary_conditions();
            
            // Periodically remesh points to maintain stability
            if step % remesh_interval == 0 {
                self.remesh_vortices();
            }
            
            // Update statistics
            if step % 10 == 0 {
                self.update_statistics_with_progress_bar(&progress_bar);
            }

            // Save time series frame if configured
            if let Some(config) = &self.time_series_config {
                if step % config.interval == 0 {
                    let frame_count = config.frame_count;
                    // progress_bar.set_message(format!("Saving frame {} at t={:.3}", frame_count, self.time));
                    if let Err(e) = self.export_time_series_frame(frame_count) {
                        eprintln!("Error saving time series frame: {}", e);
                    }
                    // Update the frame count after using it
                    if let Some(config) = &mut self.time_series_config {
                        config.frame_count += 1;
                    }
                }
            }
            
            // Save checkpoint
            if step % checkpoint_interval == 0 && step > 0 {
                let checkpoint_filename = format!("checkpoint_{}.json", step);
                
                // Update the checkpoint status bar to "Saving..."
                update_checkpoint_status(&format!("Saving: {}", checkpoint_filename));
                
                if let Err(e) = self.save_checkpoint(&checkpoint_filename) {
                    update_checkpoint_status(&format!("Error saving: {}", e));
                } else {
                    // Update with success message containing the filename
                    update_checkpoint_status(&format!("Last checkpoint: {}", checkpoint_filename));
                }
            }
            
            self.time += dt;
        }
        
        progress_bar.finish_with_message(format!(
            "\nSimulation complete! Final state: {} vortex lines, t = {:.3} s", 
            self.vortex_lines.len(), self.time
        ));
        clear_global_progress_bar();
    }

    // Add these new methods for handling progress bar updates
    fn handle_reconnections_with_progress_bar(&mut self, progress_bar: &ProgressBar) {
        let reconnection_threshold = 0.01 * self.radius;
        let previous_line_count = self.vortex_lines.len();
        let previous_total_length = physics::calculate_total_length(&self.vortex_lines);
        
        if self.using_gpu && self.compute_core.is_some() {
            // Use GPU for detecting potential reconnection points
            let compute_core = self.compute_core.as_ref().unwrap();
            let reconnection_candidates = compute_core.detect_reconnections(
                &self.vortex_lines, 
                reconnection_threshold
            );

            progress_bar.set_message(format!(
                "Found {} reconnection candidates", 
                reconnection_candidates.len()));
            
            if !reconnection_candidates.is_empty() {
                // Process the GPU-detected reconnection points
                physics::process_reconnections(&mut self.vortex_lines, reconnection_candidates);
                
                // Count as a reconnection event
                self.stats.reconnection_count += 1;
                
                // After reconnection, remove tiny loops and update line data
                physics::remove_tiny_loops(&mut self.vortex_lines, 0.005 * self.radius);
            }
        } else {
            // Existing CPU implementation
            physics::handle_reconnections(&mut self.vortex_lines, reconnection_threshold);
        }
        
        // After reconnection, check if line count or length changed significantly
        let current_line_count = self.vortex_lines.len();
        let current_total_length = physics::calculate_total_length(&self.vortex_lines);
        
        let line_count_changed = current_line_count != previous_line_count;
        let length_reduced = (previous_total_length - current_total_length) > 0.001;
        
        if line_count_changed || length_reduced {
            progress_bar.set_message(format!(
                "Reconnection at t={:.3}s: Lines {} → {}, Length {:.3} → {:.3}",
                self.time, previous_line_count, current_line_count,
                previous_total_length, current_total_length
            ));
        }
    }

    fn update_statistics_with_progress_bar(&mut self, progress_bar: &ProgressBar) {
        // Calculate total vortex length
        let total_length = physics::calculate_total_length(&self.vortex_lines);
        
        // Calculate kinetic energy
        let kinetic_energy = physics::calculate_kinetic_energy(&self.vortex_lines);
        
        // Store the values
        self.stats.total_length.push(total_length);
        self.stats.kinetic_energy.push(kinetic_energy);
        self.stats.time_points.push(self.time);
        
        // Update length status
        update_length_status(&format!("Total length: {:.4} cm", total_length));
        
        // Every 10 updates, calculate and print detailed energy breakdown
        if self.stats.time_points.len() % 10 == 0 {
            let (e_kin, e_pot, e_wave) = self.calculate_detailed_energy();
            update_energy_status(&format!(
                "E_kin={:.2e}, E_pot={:.2e}, E_wave={:.2e}", 
                e_kin, e_pot, e_wave
            ));
        }
    }

    fn initialize_vortices(&mut self) {        
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
        
        // Check if we have a GPU compute core
        if self.using_gpu && self.compute_core.is_some() {
            let compute_core = self.compute_core.as_ref().unwrap();
            
            // 1. GPU-accelerated thermal fluctuations if temperature > 0
            if self.temperature > 0.01 {
                let fluctuations = compute_core.calculate_thermal_fluctuations(
                    &self.vortex_lines,
                    self.temperature,
                    dt
                );
                
                // Apply fluctuations
                for (line_idx, line) in self.vortex_lines.iter_mut().enumerate() {
                    for (point_idx, point) in line.points.iter_mut().enumerate() {
                        let fluct = &fluctuations[line_idx][point_idx];
                        point.position[0] += fluct[0];
                        point.position[1] += fluct[1];
                        point.position[2] += fluct[2];
                    }
                }
            }
            
            // 2. GPU-accelerated velocity calculation
            let velocities = compute_core.calculate_velocities(
                &self.vortex_lines, 
                self.temperature,
                self.time,
                self.radius,
                self.height,
                self.external_field.as_ref()
            );
            
            // Apply velocities
            for (line_idx, line) in self.vortex_lines.iter_mut().enumerate() {
                for (point_idx, point) in line.points.iter_mut().enumerate() {
                    let velocity = &velocities[line_idx][point_idx];
                    point.position[0] += velocity[0] * dt;
                    point.position[1] += velocity[1] * dt;
                    point.position[2] += velocity[2] * dt;
                }
            }
        } else {
            // Add thermal fluctuations if temperature > 0
            if self.temperature > 0.01 {
                self.add_thermal_fluctuations(dt);
            }
            physics::evolve_vortex_network(
                &mut self.vortex_lines, 
                dt, 
                self.temperature,
                ext_field.as_ref(),
                self.time
            );
        }
    }

    fn add_thermal_fluctuations(&mut self, dt: f64) {
        // Only apply thermal fluctuations if temperature is above absolute zero
        if self.temperature < 0.01 {
            return;
        }
        if self.using_gpu && self.compute_core.is_some() {
            // Use GPU implementation
            let compute_core = self.compute_core.as_ref().unwrap();
            let fluctuations = compute_core.calculate_thermal_fluctuations(
                &self.vortex_lines,
                self.temperature,
                dt
            );
            
            // Apply the GPU-calculated fluctuations to vortex positions
            for (line_idx, line) in self.vortex_lines.iter_mut().enumerate() {
                for (point_idx, point) in line.points.iter_mut().enumerate() {
                    let fluct = &fluctuations[line_idx][point_idx];
                    point.position[0] += fluct[0];
                    point.position[1] += fluct[1];
                    point.position[2] += fluct[2];
                }
            }
        } else {
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
    }
    
    fn remesh_vortices(&mut self) {
        let target_spacing = self.radius / 50.0;
        let min_spacing = target_spacing * 0.5;
        let max_spacing = target_spacing * 2.0;
        
        if self.using_gpu && self.compute_core.is_some() {
            // Use GPU for remeshing
            let compute_core = self.compute_core.as_ref().unwrap();
            self.vortex_lines = compute_core.remesh_vortices_gpu(
                &self.vortex_lines,
                target_spacing,
                min_spacing,
                max_spacing
            );
        } else {
            // Existing CPU implementation
            for line in &mut self.vortex_lines {
                physics::remesh_vortex_line(line, target_spacing, min_spacing, max_spacing);
            }
        }
    }
    
    fn handle_reconnections(&mut self) {
        let reconnection_threshold = 0.01 * self.radius;
        let previous_line_count = self.vortex_lines.len();
        let previous_total_length = physics::calculate_total_length(&self.vortex_lines);
        
        if self.using_gpu && self.compute_core.is_some() {
            // Use GPU for detecting potential reconnection points
            let compute_core = self.compute_core.as_ref().unwrap();
            let reconnection_candidates = compute_core.detect_reconnections(
                &self.vortex_lines, 
                reconnection_threshold
            );
            
            if !reconnection_candidates.is_empty() {
                // Process the GPU-detected reconnection points
                physics::process_reconnections(&mut self.vortex_lines, reconnection_candidates);
                
                // Count as a reconnection event
                self.stats.reconnection_count += 1;
                
                // After reconnection, remove tiny loops and update line data
                physics::remove_tiny_loops(&mut self.vortex_lines, 0.005 * self.radius);
            }
        } else {
            // Existing CPU implementation
            physics::handle_reconnections(&mut self.vortex_lines, reconnection_threshold);
        }
        
        // After reconnection, check if line count or length changed significantly
        let current_line_count = self.vortex_lines.len();
        let current_total_length = physics::calculate_total_length(&self.vortex_lines);
        
        let line_count_changed = current_line_count != previous_line_count;
        let length_reduced = (previous_total_length - current_total_length) > 0.001;
        
        if line_count_changed || length_reduced {
            log_message(&format!(
                "Reconnection detected at t={:.3}s: Lines {} -> {}, Length {:.3} -> {:.3}",
                self.time, previous_line_count, current_line_count,
                previous_total_length, current_total_length
            ));
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
            log_message(&format!(
                "Energy: Kinetic={:.4e}, Potential={:.4e}, Waves={:.4e}", e_kin, e_pot, e_wave
            ));
        }
    }

    fn ensure_directory_exists(path: &str) -> io::Result<()> {
        let path = Path::new(path);
        if !path.exists() {
            fs::create_dir_all(path)?;
        }
        Ok(())
    }

    // Save checkpoint for resuming simulation later
    pub fn save_checkpoint(&self, filename: &str) -> io::Result<()> {
        // Ensure checkpoints directory exists
        Self::ensure_directory_exists("checkpoints")?;
        
        // Prepend the directory to the filename
        let checkpoint_path = format!("checkpoints/{}", filename);
        
        // Create the file and write the data
        let file = File::create(&checkpoint_path)?;
        serde_json::to_writer(file, self)?;
        
        Ok(())
    }
    
    // Load checkpoint to resume simulation
    pub fn load_checkpoint(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let sim: VortexSimulation = serde_json::from_reader(file)?;
        Ok(sim)
    }
    
    // Save simulation results
    pub fn save_results(&self, filename: &str) {
        // Extract the base filename without any directory parts
        let base_filename = Path::new(filename).file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
        
        // Use the detailed VTK export instead of basic save_vtk
        if let Err(e) = self.save_detailed_vtk(&base_filename) {
            eprintln!("Error saving VTK file: {}", e);
        }
        
        // Save statistics to JSON
        let stats_filename = base_filename.replace(".vtk", "_stats.json");
        if let Err(e) = self.save_statistics(&stats_filename) {
            eprintln!("Error saving statistics: {}", e);
        }
    }

    // New method to save results for time series visualization
    pub fn save_time_series_vtk(&mut self, base_filename: &str, interval: usize) {
        // Store this function to be called during simulation
        self.time_series_config = Some(TimeSeriesConfig {
            base_filename: base_filename.to_string(),
            interval,
            frame_count: 0,
        });
    }
    
    // Helper method to export a single frame of the time series
    fn export_time_series_frame(&self, frame_number: usize) -> io::Result<()> {
        if let Some(config) = &self.time_series_config {
            // Ensure visualization directory exists
            Self::ensure_directory_exists("visualization")?;
            
            // Create filename with frame number
            let filename = format!("visualization/{}_{:04}.vtk", config.base_filename, frame_number);
            
            // Save VTK with time field explicitly set - use the direct path here
            let mut file = File::create(&filename)?;
            
            // VTK header
            writeln!(file, "# vtk DataFile Version 3.0")?;
            writeln!(file, "Superfluid Helium Vortex Tangle Simulation t={:.4}", self.time)?;
            writeln!(file, "ASCII")?;
            writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
            
            // Important: Add FIELD data BEFORE other data to ensure proper loading
            writeln!(file, "FIELD FieldData 1")?;
            writeln!(file, "TIME 1 1 double")?;
            writeln!(file, "{}", self.time)?;
            
            // Write all the geometric and scalar data
            self.write_vtk_data(&mut file)?;
            
            // Also generate a ParaView state file for the first frame
            if frame_number == 0 {
                self.generate_paraview_state(&format!("visualization/{}", config.base_filename))?;
            }
        }
        Ok(())
    }
    
    pub fn save_detailed_vtk(&self, filename: &str) -> io::Result<()> {
        // Simply call save_detailed_vtk_with_time with the current simulation time
        self.save_detailed_vtk_with_time(filename, self.time)
    }
    
    fn save_detailed_vtk_with_time(&self, filename: &str, time: f64) -> io::Result<()> {
        // Ensure visualization directory exists
        Self::ensure_directory_exists("visualization")?;
        
        // Extract the base filename without any directory parts
        let base_filename = Path::new(filename).file_name()
                                 .unwrap_or_default()
                                 .to_string_lossy()
                                 .to_string();
        
        // Prepend the directory to the filename
        let vis_path = format!("visualization/{}", base_filename);
        
        let mut file = File::create(&vis_path)?;
        
        // VTK header
        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "Superfluid Helium Vortex Tangle Simulation t={:.4}", time)?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        
        // Important: Add FIELD data BEFORE other data to ensure proper loading
        writeln!(file, "FIELD FieldData 1")?;
        writeln!(file, "TIME 1 1 double")?;
        writeln!(file, "{}", time)?;
        
        // Write all the geometric and scalar data
        self.write_vtk_data(&mut file)?;
        
        Ok(())
    }
    
    // Helper method to write the VTK data - remove TIME field from this method
    fn write_vtk_data(&self, file: &mut File) -> io::Result<()> {
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
            if n_points > 1 {  // Only create cells if there are at least 2 points
                total_cells += n_points - 1;  // segments between points
                total_list_size += (n_points - 1) * 3;  // 3 values per segment (type + 2 points)
            }
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
        
        // Only write point data if there are points
        if total_points > 0 {
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
        }
        
        Ok(())
    }
    
    // Generate a simple ParaView state file for easy loading
    fn generate_paraview_state(&self, base_filename: &str) -> io::Result<()> {
        // Create a README file with instructions
        let readme_path = format!("{}_README.txt", base_filename);
        let mut file = File::create(&readme_path)?;
        
        // Write README contents
        writeln!(file, "# Vortex Simulation Time Series Visualization")?;
        writeln!(file, "==================================")?;
        writeln!(file, "")?;
        writeln!(file, "## Loading in ParaView")?;
        // ... rest of README content

        // Create a Python script for advanced users
        let py_path = format!("{}_load.py", base_filename);
        let mut py_file = File::create(&py_path)?;
        
        // Write Python script contents
        writeln!(py_file, "# ParaView Python script to load the time series")?;
        writeln!(py_file, "import os")?;
        writeln!(py_file, "import glob")?;
        writeln!(py_file, "from paraview.simple import *")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Disable automatic rendering during script execution")?;
        writeln!(py_file, "paraview.simple._DisableFirstRenderCameraReset()")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Close any existing rendering views")?;
        writeln!(py_file, "try:")?;
        writeln!(py_file, "    Disconnect()")?;
        writeln!(py_file, "    Connect()")?;
        writeln!(py_file, "except:")?;
        writeln!(py_file, "    pass")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Get the directory of this script")?;
        writeln!(py_file, "script_dir = os.path.dirname(os.path.abspath(__file__))")?;
        writeln!(py_file, "file_pattern = os.path.join(script_dir, '{}_*.vtk')", base_filename)?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Find all matching files and sort them")?;
        writeln!(py_file, "files = sorted(glob.glob(file_pattern))")?;
        writeln!(py_file, "if not files:")?;
        writeln!(py_file, "    print(f\"Error: No files found matching {{file_pattern}}\")")?;
        writeln!(py_file, "    exit(1)")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "print(f\"Loading {{len(files)}} time series files...\")")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Load the data")?;
        writeln!(py_file, "reader = LegacyVTKReader(registrationName=\"VortexTimeSeriesReader\", FileNames=files)")?;
        writeln!(py_file, "reader.UpdatePipeline()")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Create a default render view")?;
        writeln!(py_file, "renderView = CreateView('RenderView')")?;
        writeln!(py_file, "renderView.ViewSize = [1000, 800]")?;
        writeln!(py_file, "renderView.AxesGrid = 'Grid Axes 3D Actor'")?;
        writeln!(py_file, "renderView.CenterOfRotation = [0.0, 0.0, {:.1}]", self.height/2.0)?;
        writeln!(py_file, "renderView.StereoType = 0")?;
        writeln!(py_file, "renderView.CameraPosition = [0.0, -3.0 * {:.1}, {:.1}]", self.radius, self.height/2.0)?;
        writeln!(py_file, "renderView.CameraFocalPoint = [0.0, 0.0, {:.1}]", self.height/2.0)?;
        writeln!(py_file, "renderView.CameraViewUp = [0.0, 0.0, 1.0]")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Create a nice visualization")?;
        writeln!(py_file, "tube = Tube(Input=reader, registrationName=\"VortexTubes\")")?;
        writeln!(py_file, "tube.Radius = {:.3}", self.radius * 0.02)?;
        writeln!(py_file, "tube.NumberOfSides = 12")?;
        writeln!(py_file, "tube.CappingOn = 0")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Display properties")?;
        writeln!(py_file, "tubeDisplay = Show(tube, renderView, 'GeometryRepresentation')")?;
        writeln!(py_file, "ColorBy(tubeDisplay, ('POINTS', 'curvature'))")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Color scale")?;
        writeln!(py_file, "curvatureLUT = GetColorTransferFunction('curvature')")?;
        writeln!(py_file, "curvatureLUT.RescaleTransferFunction(0, 10.0)")?;
        writeln!(py_file, "try:")?;
        writeln!(py_file, "    curvatureLUT.ApplyPreset('Viridis (matplotlib)', True)")?;
        writeln!(py_file, "except:")?;
        writeln!(py_file, "    pass")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Show color legend")?;
        writeln!(py_file, "curvatureLUTColorBar = GetScalarBar(curvatureLUT, renderView)")?;
        writeln!(py_file, "curvatureLUTColorBar.Title = 'Curvature'")?;
        writeln!(py_file, "curvatureLUTColorBar.ComponentTitle = ''")?;
        writeln!(py_file, "tubeDisplay.SetScalarBarVisibility(renderView, True)")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Set up animation")?;
        writeln!(py_file, "animationScene = GetAnimationScene()")?;
        writeln!(py_file, "animationScene.PlayMode = 'Snap To TimeSteps'")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "# Reset view to fit data")?;
        writeln!(py_file, "Render()")?;
        writeln!(py_file, "ResetCamera()")?;
        writeln!(py_file, "")?;
        writeln!(py_file, "print(\"Time series loaded successfully. Press Play in the animation controls to view.\")")?;
        
        Ok(())
    }
    
    // Save detailed statistics
    // Save detailed statistics
    fn save_statistics(&self, filename: &str) -> io::Result<()> {
        // Ensure checkpoints directory exists
        Self::ensure_directory_exists("checkpoints")?;
        
        // Extract the base filename without any directory parts
        let base_filename = Path::new(filename).file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
        
        // Prepend the directory to the filename
        let stats_path = format!("checkpoints/{}", base_filename);
        
        let file = File::create(&stats_path)?;
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
                wavelengths
            );
        }
    }
}

pub fn run_parameter_study(
    base_radius: f64,
    height: f64,
    base_temperature: f64,
    radius_range: (f64, f64, usize),
    temp_range: (f64, f64, usize),
    wave_amplitude: f64,
    steps: usize,
    output_dir: &str,
    compute_core: Option<ComputeCore>,
    external_field: Option<ExternalFieldParams>,
) -> Vec<SimulationResult> {
    let (radius_min, radius_max, radius_steps) = radius_range;
    let (temp_min, temp_max, temp_steps) = temp_range;
    
    let radius_values = generate_parameter_range(radius_min, radius_max, radius_steps);
    let temp_values = generate_parameter_range(temp_min, temp_max, temp_steps);
    
    let total_runs = radius_values.len() * temp_values.len();
    log_message(&format!("Running {} simulations in total", total_runs));
    
    let mut results = Vec::new();
    let mut run_count = 0;
    
    // Create progress bar
    let multi_progress = MultiProgress::new();
        let progress_bar = multi_progress.add(ProgressBar::new(steps as u64));
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-")
        );
        set_global_progress_bar(multi_progress, progress_bar.clone());
    
    // Reuse compute core across simulations if provided
    let shared_compute_core = compute_core;
    
    for radius in &radius_values {
        for temp in &temp_values {
            run_count += 1;
            progress_bar.set_message(format!(
                "Radius: {:.2} cm, Temp: {:.2} K", radius, temp));
            
            // Create simulation with optional GPU core
            let mut sim = if let Some(ref core) = shared_compute_core {
                let cloned_core = core.clone();
                VortexSimulation::new_with_compute_core(*radius, height, *temp, cloned_core)
            } else {
                VortexSimulation::new(*radius, height, *temp)
            };
            
            // Add external field if provided
            if let Some(ref field) = external_field {
                sim.external_field = Some(field.clone());
            }
            
            // Add Kelvin waves and run simulation
            if wave_amplitude > 0.0 {
                sim.add_kelvin_waves(wave_amplitude * sim.radius, 3.0);
            }
            
            sim.run(steps);
            
            // Save individual result
            let output_file = format!("{}/sim_r{:.2}_t{:.2}.vtk", 
                                     output_dir, radius, temp);
            sim.save_results(&output_file);
            
            // Collect metrics
            let result = SimulationResult {
                radius: *radius,
                height,
                temperature: *temp,
                wave_amplitude,
                steps,
                total_length: sim.stats.total_length.last().copied().unwrap_or(0.0),
                energy: sim.stats.kinetic_energy.last().copied().unwrap_or(0.0),
                reconnection_count: sim.stats.reconnection_count,
            };
            
            results.push(result);
            progress_bar.inc(1);
        }
    }
    
    progress_bar.finish_with_message("Parameter study complete!");
    println!("Finished running {} simulations", run_count);
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

pub fn parse_external_field(
    field_type: &str,
    value_str: &str,
    center_str: &str,
    frequency: f64,
    phase: f64,
) -> Result<Option<ExternalFieldParams>, Box<dyn std::error::Error>> {
    // Parse vector values
    fn parse_vec3(s: &str) -> Result<[f64; 3], Box<dyn std::error::Error>> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err("Vector format should be 'x,y,z'".into());
        }
        
        Ok([
            parts[0].trim().parse::<f64>()?,
            parts[1].trim().parse::<f64>()?,
            parts[2].trim().parse::<f64>()?,
        ])
    }

    match field_type.to_lowercase().as_str() {
        "none" => Ok(None),
        
        "rotation" => {
            let angular_velocity = parse_vec3(value_str)?;
            let center = parse_vec3(center_str)?;
            Ok(Some(ExternalFieldParams::Rotation { 
                angular_velocity,
                center
            }))
        },
        
        "uniform" => {
            let velocity = parse_vec3(value_str)?;
            Ok(Some(ExternalFieldParams::UniformFlow { velocity }))
        },
        
        "oscillatory" => {
            let amplitude = parse_vec3(value_str)?;
            Ok(Some(ExternalFieldParams::OscillatoryFlow { 
                amplitude,
                frequency, 
                phase
            }))
        },
        
        "counterflow" => {
            let velocity = parse_vec3(value_str)?;
            Ok(Some(ExternalFieldParams::Counterflow { velocity }))
        },
        
        _ => Err(format!("Unknown external field type: {}", field_type).into())
    }
}

// MARK: Simulation Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub radius: f64,
    pub height: f64,
    pub temperature: f64,
    pub wave_amplitude: f64,
    pub steps: usize,
    pub total_length: f64,
    pub energy: f64,
    pub reconnection_count: usize,
}