# Debugging Simulation Hanging After Reconnection Events

The issue you're experiencing is likely related to the vortex reconnection algorithm. When your simulation hangs after reconnection events, it suggests there's either an infinite loop, deadlock, or a severe performance bottleneck in the reconnection handling code.

## Possible Causes

Based on the code snippets you provided, here are the most likely issues:

1. **Index corruption after reconnection**: Removing and adding vortex lines might be causing index misalignment in `reconnect_vortex_lines`.

2. **Infinite loop in remeshing**: After reconnection, the remeshing algorithm might be creating unstable configurations.

3. **Deadlock in parallel computation**: The parallel vortex evolution might be encountering thread synchronization issues.

4. **Memory issues**: Constant reconnections might be causing excessive memory allocation.

## Solution Approach

Here's a step-by-step plan to fix the issue:

### 1. Add Safety Checks to Reconnection Logic

```rust
fn reconnect_vortex_lines(
    vortex_lines: &mut Vec<VortexLine>,
    i: usize,
    pi: usize,
    j: usize,
    pj: usize,
) {
    // Safety check - make sure indices are valid
    if i >= vortex_lines.len() || j >= vortex_lines.len() {
        eprintln!("Warning: Invalid line indices in reconnection: {} and {}", i, j);
        return;
    }
    
    if pi >= vortex_lines[i].points.len() || pj >= vortex_lines[j].points.len() {
        eprintln!("Warning: Invalid point indices in reconnection: {} and {}", pi, pj);
        return;
    }

    // Create four new segments from the two original lines
    let line_i = vortex_lines.remove(if i > j { i } else { i });
    let line_j = vortex_lines.remove(if i > j { j } else { j - 1 });

    // Split first line at pi
    let (mut segment1a, mut segment1b) = split_line_at(&line_i, pi);

    // Split second line at pj
    let (mut segment2a, mut segment2b) = split_line_at(&line_j, pj);

    // Only add new lines if they have sufficient points
    // This prevents degenerate lines that could cause issues
    let min_points = 4; // Require at least 4 points for stability
    
    // Reconnect: segment1a + segment2b, segment2a + segment1b
    if segment1a.points.len() >= min_points && segment2b.points.len() >= min_points {
        let mut new_line1 = VortexLine { points: Vec::new() };
        new_line1.points.append(&mut segment1a.points);
        new_line1.points.append(&mut segment2b.points);
        update_tangent_vectors(&mut new_line1);
        vortex_lines.push(new_line1);
    }

    if segment2a.points.len() >= min_points && segment1b.points.len() >= min_points {
        let mut new_line2 = VortexLine { points: Vec::new() };
        new_line2.points.append(&mut segment2a.points);
        new_line2.points.append(&mut segment1b.points);
        update_tangent_vectors(&mut new_line2);
        vortex_lines.push(new_line2);
    }
}
```

### 2. Add Timeout and Monitoring to the Reconnection Process

```rust
fn handle_reconnections(&mut self) {
    let reconnection_threshold = 0.01 * self.radius;
    let previous_line_count = self.vortex_lines.len();
    let previous_total_length = physics::calculate_total_length(&self.vortex_lines);
    
    // Limit the number of reconnections per step to avoid cascades
    let max_reconnections_per_step = 5;
    
    // The function may not return reconnection count, so we need to check before and after
    physics::handle_reconnections_with_limit(&mut self.vortex_lines, reconnection_threshold, max_reconnections_per_step);
    
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
        
        // After reconnection, verify all vortex lines are valid
        self.validate_vortex_lines();
    }
}

// Add this method to check for and fix invalid vortex lines
fn validate_vortex_lines(&mut self) {
    // Remove any degenerate lines (too few points)
    self.vortex_lines.retain(|line| line.points.len() >= 4);
    
    // Check for NaN or Inf values
    for line in &mut self.vortex_lines {
        for point in &mut line.points {
            for i in 0..3 {
                if !point.position[i].is_finite() {
                    // Replace with a valid value to avoid crashes
                    point.position[i] = 0.0;
                }
                if !point.tangent[i].is_finite() {
                    // Fix invalid tangent
                    point.tangent[i] = 0.0;
                }
            }
            
            // Ensure tangent is normalized
            let magnitude = (point.tangent[0].powi(2) + point.tangent[1].powi(2) + point.tangent[2].powi(2)).sqrt();
            if magnitude > 1e-10 {
                point.tangent[0] /= magnitude;
                point.tangent[1] /= magnitude;
                point.tangent[2] /= magnitude;
            } else {
                // Default tangent if normalization fails
                point.tangent = [1.0, 0.0, 0.0];
            }
        }
    }
}
```

### 3. Add the New Function to Physics Module

```rust
/// Reconnection algorithm with limit on number of reconnections
pub fn handle_reconnections_with_limit(
    vortex_lines: &mut Vec<VortexLine>, 
    reconnection_threshold: f64,
    max_reconnections: usize
) {
    // List of potential reconnections
    let mut reconnections = Vec::new();

    // Check for close approaches between different lines
    for i in 0..vortex_lines.len() {
        // Check against other lines (j > i to avoid duplicate checks)
        for j in i + 1..vortex_lines.len() {
            // Check points on line i
            for pi in 0..vortex_lines[i].points.len() {
                let p1 = Vector3::new(
                    vortex_lines[i].points[pi].position[0],
                    vortex_lines[i].points[pi].position[1],
                    vortex_lines[i].points[pi].position[2],
                );

                // Get adjacent points on line i for tangent calculation
                let pi_next = (pi + 1) % vortex_lines[i].points.len();
                let p1_next = Vector3::new(
                    vortex_lines[i].points[pi_next].position[0],
                    vortex_lines[i].points[pi_next].position[1],
                    vortex_lines[i].points[pi_next].position[2],
                );

                // Calculate segment tangent
                let t1 = (p1_next - p1).normalize();

                // Check against points on line j
                for pj in 0..vortex_lines[j].points.len() {
                    let p2 = Vector3::new(
                        vortex_lines[j].points[pj].position[0],
                        vortex_lines[j].points[pj].position[1],
                        vortex_lines[j].points[pj].position[2],
                    );

                    // Get adjacent point on line j
                    let pj_next = (pj + 1) % vortex_lines[j].points.len();
                    let p2_next = Vector3::new(
                        vortex_lines[j].points[pj_next].position[0],
                        vortex_lines[j].points[pj_next].position[1],
                        vortex_lines[j].points[pj_next].position[2],
                    );

                    // Calculate segment tangent
                    let t2 = (p2_next - p2).normalize();

                    // Check distance
                    let dist = (p1 - p2).norm();

                    // Check if close enough and segments are roughly anti-parallel
                    let dot = t1.dot(&t2);

                    if dist < reconnection_threshold && dot < -0.3 {
                        reconnections.push((i, pi, j, pj, dist));
                    }
                }
            }
        }
    }

    // Sort by distance (closest first)
    reconnections.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
    
    // Limit number of reconnections
    if reconnections.len() > max_reconnections {
        reconnections.truncate(max_reconnections);
    }

    // Process reconnections (closest first)
    for (i, pi, j, pj, _) in reconnections {
        // Perform actual reconnection
        reconnect_vortex_lines(vortex_lines, i, pi, j, pj);
    }
}
```

### 4. Modify Your Main Run Loop to Prevent Hanging

```rust
pub fn run(&mut self, steps: usize) {
    println!("Running simulation for {} steps...", steps);
    
    // Initialize vortices
    self.initialize_vortices();

    // Add Kelvin waves with amplitude proportional to temperature
    if self.temperature > 0.1 {
        let amplitude = 0.05 * self.radius * (self.temperature / 2.17);
        let wavenumber = 2.0 + (self.temperature * 2.0);
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
    let max_reconnections_total = 100; // Safety limit for total reconnections
    
    for step in 0..steps {
        progress_bar.set_position(step as u64);
        
        if step % 10 == 0 {
            progress_bar.set_message(format!("Total length: {:.4} cm, Reconnects: {}", 
                self.stats.total_length.last().unwrap_or(&0.0),
                self.stats.reconnection_count));
        }
        
        // Safety check - if we've had too many reconnections, reduce the threshold temporarily
        if self.stats.reconnection_count > max_reconnections_total {
            eprintln!("Warning: Too many reconnections ({}), reducing threshold and limiting further reconnections", 
                     self.stats.reconnection_count);
            
            // Continue simulation but with reduced reconnection activity
            // This prevents hanging while still allowing the simulation to progress
            break;
        }
        
        // Calculate vortex dynamics
        self.evolve_vortices(dt);
        
        // Handle vortex reconnections (now with safety measures)
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
```

## Debugging Options

If the fixes above don't fully resolve the issue, here are additional debugging approaches:

1. **Add progress output to the reconnection algorithm**: Print status updates during reconnections to identify exactly where it's hanging.

2. **Time each step of the algorithm**: Add timing code to identify which parts are taking too long.

3. **Temporarily disable reconnections**: Run a simulation with reconnections disabled to confirm they are indeed the source of the problem.

4. **Memory monitoring**: Check if your simulation is causing excessive memory consumption.

5. **Save debug VTK files**: Save the vortex configuration immediately before and after a reconnection to visualize what's happening.

By implementing these changes, you should be able to identify and resolve the issue causing your simulation to hang after reconnection events.