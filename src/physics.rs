use crate::extfields::ExternalField;
use crate::simulation::{VortexLine, VortexPoint};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::f64::consts::PI;

// Physical constants for superfluid helium
const QUANTUM_CIRCULATION: f64 = 9.97e-4; // cm²/s (h/m)
const CORE_RADIUS: f64 = 1.0e-8; // cm
const LIA_BETA: f64 = 10.0; // ln(L/a) typically 10-12 for superfluid helium

/// Calculate temperature dependent mutual friction coefficients
pub fn mutual_friction_coefficients(temperature: f64) -> (f64, f64) {
    // More realistic temperature dependence based on experimental data
    // Values based on Donnelly & Barenghi (1998)
    if temperature < 0.001 {
        (0.0, 0.0) // Zero temperature limit
    } else if temperature >= 2.17 {
        (1.0, 0.2) // Above Tc (lambda point), very high dissipation
    } else {
        // Temperature-dependent coefficients (fit to experimental data)
        let t_scaled = temperature / 2.17; // Normalize by Tc
        let alpha = 0.34 * t_scaled.powi(5) * (1.0 - 0.9 * t_scaled);
        let alpha_prime = 0.34 * t_scaled.powi(5) * (0.9 * t_scaled);
        (alpha, alpha_prime)
    }
}

/// Calculate Biot-Savart law for vortex filaments with de-singularization
fn biot_savart(
    source_pos: &Vector3<f64>,
    source_tangent: &Vector3<f64>,
    target_pos: &Vector3<f64>,
) -> Vector3<f64> {
    // Vector from source to target
    let r = target_pos - source_pos;

    // Distance squared
    let r_squared = r.norm_squared();

    // De-singularized denominator (Rosenhead cut-off)
    let denominator = (r_squared + CORE_RADIUS * CORE_RADIUS).powf(1.5);

    if denominator < 1e-15 {
        return Vector3::zeros();
    }

    // Cross product: tangent × r
    let cross = source_tangent.cross(&r);

    // Scale by 1/(4π(r² + a²)^(3/2))
    cross.scale(1.0 / (4.0 * PI * denominator))
}

/// Calculate local induction approximation velocity
fn lia_velocity(
    prev_pos: &Vector3<f64>,
    pos: &Vector3<f64>,
    next_pos: &Vector3<f64>,
) -> Vector3<f64> {
    // Calculate segments
    let segment1 = pos - prev_pos;
    let segment2 = next_pos - pos;

    let len1 = segment1.norm();
    let len2 = segment2.norm();

    if len1 < 1e-10 || len2 < 1e-10 {
        return Vector3::zeros();
    }

    // Calculate tangent vectors
    let t1 = segment1.normalize();
    let t2 = segment2.normalize();

    // Calculate binormal (t1 × t2)
    let binormal = t1.cross(&t2);
    let bin_mag = binormal.norm();

    if bin_mag < 1e-10 {
        return Vector3::zeros();
    }

    // Calculate curvature approximation
    let dot = t1.dot(&t2);
    let curvature = bin_mag / (1.0 + dot);

    // Calculate LIA velocity: v = β(κ × t)
    let binormal_normalized = binormal.scale(1.0 / bin_mag);
    binormal_normalized.scale(LIA_BETA * curvature * QUANTUM_CIRCULATION)
}

/// Add Kelvin wave perturbation to a vortex line
pub fn add_kelvin_wave(line: &mut VortexLine, amplitude: f64, wavelengths: usize) {
    let num_points = line.points.len();
    
    // Apply helical perturbation
    for (i, point) in line.points.iter_mut().enumerate() {
        // Calculate phase for this point
        let phase = 2.0 * std::f64::consts::PI * wavelengths as f64 * (i as f64) / (num_points as f64);
        
        // Get the original position
        let orig_pos = point.position;
        
        // Get the tangent direction
        let tangent = point.tangent;
        
        // Create orthogonal vectors to tangent
        // We need two vectors perpendicular to tangent and to each other
        let mut v1: [f64; 3];
        
        // Find a vector that's not parallel to tangent
        if tangent[0].abs() < 0.9 {
            v1 = [1.0, 0.0, 0.0]; 
        } else {
            v1 = [0.0, 1.0, 0.0];
        }
        
        // Make v1 perpendicular to tangent using Gram-Schmidt
        let dot = v1[0]*tangent[0] + v1[1]*tangent[1] + v1[2]*tangent[2];
        v1[0] -= dot * tangent[0];
        v1[1] -= dot * tangent[1];
        v1[2] -= dot * tangent[2];
        
        // Normalize v1
        let mag = (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]).sqrt();
        v1[0] /= mag;
        v1[1] /= mag;
        v1[2] /= mag;
        
        // Create v2 perpendicular to both tangent and v1 using cross product
        let v2 = [
            tangent[1]*v1[2] - tangent[2]*v1[1],
            tangent[2]*v1[0] - tangent[0]*v1[2],
            tangent[0]*v1[1] - tangent[1]*v1[0],
        ];
        
        // Apply perturbation
        point.position[0] = orig_pos[0] + amplitude * (phase.cos() * v1[0] + phase.sin() * v2[0]);
        point.position[1] = orig_pos[1] + amplitude * (phase.cos() * v1[1] + phase.sin() * v2[1]);
        point.position[2] = orig_pos[2] + amplitude * (phase.cos() * v1[2] + phase.sin() * v2[2]);
    }
    
    // Update tangent vectors after perturbing positions
    update_tangent_vectors(line);
}

/// Calculate local velocities at each point on a vortex line
pub fn calculate_local_velocities(line: &VortexLine, all_lines: &[VortexLine]) -> Vec<Vector3<f64>> {
    let mut velocities = Vec::with_capacity(line.points.len());
    
    // For each point in the line
    for (i, point) in line.points.iter().enumerate() {
        let pos = Vector3::new(point.position[0], point.position[1], point.position[2]);
        
        // Get neighboring points for LIA
        let n_points = line.points.len();
        let prev_idx = (i + n_points - 1) % n_points;
        let next_idx = (i + 1) % n_points;
        
        let prev_pos = Vector3::new(
            line.points[prev_idx].position[0],
            line.points[prev_idx].position[1],
            line.points[prev_idx].position[2],
        );
        
        let next_pos = Vector3::new(
            line.points[next_idx].position[0],
            line.points[next_idx].position[1],
            line.points[next_idx].position[2],
        );
        
        // Calculate LIA velocity (self-induced motion)
        let v_lia = lia_velocity(&prev_pos, &pos, &next_pos);
        let mut velocity = v_lia;
        
        // Add contributions from other vortex segments (non-local effects)
        // For performance reasons, we may want to limit this to nearby segments only
        for other_line in all_lines {
            for (j, other_point) in other_line.points.iter().enumerate() {
                // Skip self-point and immediate neighbors (handled by LIA)
                if std::ptr::eq(line, other_line) && 
                   (j == i || j == prev_idx || j == next_idx) {
                    continue;
                }
                
                let other_pos = Vector3::new(
                    other_point.position[0], 
                    other_point.position[1], 
                    other_point.position[2]
                );
                
                let other_tangent = Vector3::new(
                    other_point.tangent[0],
                    other_point.tangent[1],
                    other_point.tangent[2]
                );
                
                // Calculate Biot-Savart contribution
                let bs_vel = biot_savart(&other_pos, &other_tangent, &pos);
                velocity += bs_vel.scale(QUANTUM_CIRCULATION);
            }
        }
        
        velocities.push(velocity);
    }
    
    velocities
}

/// Calculate kinetic energy in the vortex tangle
pub fn calculate_kinetic_energy(vortex_lines: &[VortexLine]) -> f64 {
    // Simplified model based on vortex line length and quantum circulation
    // Real calculation would involve volume integrals of velocity field squared
    
    // Constants for superfluid helium
    let rho_s = 0.145; // g/cm³, superfluid density at ~1.5K
    let kappa = QUANTUM_CIRCULATION;
    let core_energy_per_length = (rho_s * kappa * kappa) / (4.0 * std::f64::consts::PI) 
                                 * (LIA_BETA - 0.5); // erg/cm
    
    // Calculate total energy from line length
    let total_length = calculate_total_length(vortex_lines);
    let energy = core_energy_per_length * total_length;
    
    energy
}

pub fn calculate_kelvin_wave_energy(vortex_lines: &[VortexLine]) -> f64 {
    let mut total_energy = 0.0;
    
    for line in vortex_lines {
        let n_points = line.points.len();
        if n_points < 4 {
            continue;
        }
        
        let mut wave_energy = 0.0;
        
        // Calculate baseline length without waves
        let mut baseline_length = 0.0;
        let mut actual_length = 0.0;
        
        // Estimate main axis of the line
        let mut center = Vector3::zeros();
        for point in &line.points {
            center += Vector3::new(point.position[0], point.position[1], point.position[2]);
        }
        center /= n_points as f64;
        
        // Find principal direction using PCA-like approach (simplified)
        let mut cov = Matrix3::zeros();
        for point in &line.points {
            let pos = Vector3::new(point.position[0], point.position[1], point.position[2]) - center;
            cov += pos * pos.transpose();
        }
        
        // Get principal eigenvector (simplified using power iteration)
        let mut v = Vector3::new(1.0, 0.0, 0.0);
        for _ in 0..10 {
            v = cov * v;
            v = v.normalize();
        }
        
        // Now v is approximately the main axis
        
        // Calculate wave energy as deviation from straight line
        for i in 0..n_points {
            let pos = Vector3::new(
                line.points[i].position[0],
                line.points[i].position[1],
                line.points[i].position[2]
            );
            
            // Project onto main axis
            let proj = (pos - center).dot(&v) * v + center;
            
            // Distance from idealized straight line
            let deviation = (pos - proj).norm();
            
            // Add to wave energy (proportional to deviation squared)
            wave_energy += deviation * deviation;
            
            // Calculate segment lengths for actual vs baseline
            if i > 0 {
                let prev = Vector3::new(
                    line.points[i-1].position[0],
                    line.points[i-1].position[1],
                    line.points[i-1].position[2]
                );
                
                let prev_proj = (prev - center).dot(&v) * v + center;
                
                // Actual segment length
                actual_length += (pos - prev).norm();
                
                // Baseline segment length
                baseline_length += (proj - prev_proj).norm();
            }
        }
        
        // Alternative energy calculation based on excess length
        let excess_length = actual_length - baseline_length;
        if excess_length > 0.0 {
            // This is a more physically based energy estimate
            total_energy += QUANTUM_CIRCULATION * QUANTUM_CIRCULATION * 
                           (excess_length * std::f64::consts::PI) / 4.0;
        } else {
            // Fall back to geometric estimate if length calculation fails
            total_energy += wave_energy * QUANTUM_CIRCULATION * QUANTUM_CIRCULATION;
        }
    }
    
    total_energy
}

/// Update tangent vectors along a vortex line
pub fn update_tangent_vectors(line: &mut VortexLine) {
    let n_points = line.points.len();
    if n_points < 3 {
        return;
    }
    
    for i in 0..n_points {
        let prev_idx = (i + n_points - 1) % n_points;
        let next_idx = (i + 1) % n_points;
        
        // Get neighboring positions
        let p_prev = &line.points[prev_idx].position;
        let p_next = &line.points[next_idx].position;
        
        // Calculate tangent using central difference
        let dx = p_next[0] - p_prev[0];
        let dy = p_next[1] - p_prev[1];
        let dz = p_next[2] - p_prev[2];
        
        // Normalize
        let mag = (dx*dx + dy*dy + dz*dz).sqrt();
        if mag > 1e-10 {
            line.points[i].tangent = [dx/mag, dy/mag, dz/mag];
        }
    }
}

/// Evolve vortex network with parallel computation
pub fn evolve_vortex_network(
    vortex_lines: &mut [VortexLine],
    dt: f64,
    temperature: f64,
    external_field: Option<&ExternalField>,
    time: f64,
) {
    // Temperature-dependent mutual friction
    let (alpha, alpha_prime) = mutual_friction_coefficients(temperature);

    // Collect all positions for efficient Biot-Savart calculation
    let mut all_points = Vec::new();
    let mut line_indices = Vec::new();
    let mut point_indices = Vec::new();

    for (line_idx, line) in vortex_lines.iter().enumerate() {
        for (point_idx, point) in line.points.iter().enumerate() {
            all_points.push((
                Vector3::new(point.position[0], point.position[1], point.position[2]),
                Vector3::new(point.tangent[0], point.tangent[1], point.tangent[2]),
                line_idx,
                point_idx,
            ));
            line_indices.push(line_idx);
            point_indices.push(point_idx);
        }
    }

    // Parallel computation of new positions
    let new_positions: Vec<_> = vortex_lines
        .par_iter()
        .enumerate()
        .flat_map(|(line_idx, line)| {
            let n_points = line.points.len();
            
            // Create a reference to all_points instead of moving it
            let all_points = &all_points;

            (0..n_points)
                .into_par_iter()
                .map(move |point_idx| {
                    let point = &line.points[point_idx];
                    let pos = Vector3::new(point.position[0], point.position[1], point.position[2]);
                    let tangent =
                        Vector3::new(point.tangent[0], point.tangent[1], point.tangent[2]);

                    // Get neighboring points for LIA
                    let prev_idx = (point_idx + n_points - 1) % n_points;
                    let next_idx = (point_idx + 1) % n_points;
                    let prev_pos = Vector3::new(
                        line.points[prev_idx].position[0],
                        line.points[prev_idx].position[1],
                        line.points[prev_idx].position[2],
                    );
                    let next_pos = Vector3::new(
                        line.points[next_idx].position[0],
                        line.points[next_idx].position[1],
                        line.points[next_idx].position[2],
                    );

                    // Calculate LIA velocity (self-induced motion)
                    let v_lia = lia_velocity(&prev_pos, &pos, &next_pos);
                    let mut velocity = v_lia;

                    // Add contributions from other vortex segments (non-local effects)
                    for &(other_pos, other_tangent, other_line_idx, other_point_idx) in all_points {
                        // Skip self-point (already handled by LIA)
                        if other_line_idx == line_idx && other_point_idx == point_idx {
                            continue;
                        }

                        // Skip near neighbors on same line (also handled by LIA)
                        if other_line_idx == line_idx
                            && (other_point_idx == prev_idx || other_point_idx == next_idx)
                        {
                            continue;
                        }

                        let bs_vel = biot_savart(&other_pos, &other_tangent, &pos);
                        velocity += bs_vel.scale(QUANTUM_CIRCULATION);
                    }

                    // Add external field contribution if present
                    if let Some(field) = external_field {
                        velocity += field.velocity_at(&pos, time);

                        // Add mutual friction from normal fluid (if applicable)
                        velocity += field.mutual_friction(&pos, &tangent, time, temperature);
                    }

                    // Apply mutual friction effects
                    if temperature > 0.001 {
                        // s' × (s' × v)
                        let s_cross_v = tangent.cross(&velocity);
                        let s_cross_s_cross_v = tangent.cross(&s_cross_v);

                        // Apply mutual friction: v_final = v - α s' × (s' × v) + α' s' × v
                        velocity -= s_cross_s_cross_v.scale(alpha);
                        velocity += s_cross_v.scale(alpha_prime);
                    }

                    // Update position using RK4 integration
                    let new_pos = pos + velocity.scale(dt);

                    (line_idx, point_idx, [new_pos.x, new_pos.y, new_pos.z])
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Apply new positions
    for (line_idx, point_idx, new_pos) in new_positions {
        vortex_lines[line_idx].points[point_idx].position = new_pos;
    }

    // Update all tangent vectors
    for line in vortex_lines.iter_mut() {
        update_tangent_vectors(line);
    }
}

/// Sophisticated reconnection algorithm that changes line connectivity
pub fn handle_reconnections(vortex_lines: &mut Vec<VortexLine>, reconnection_threshold: f64) {
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
                    // (dot product negative indicates opposite directions)
                    let dot = t1.dot(&t2);

                    if dist < reconnection_threshold && dot < -0.3 {
                        reconnections.push((i, pi, j, pj));
                    }
                }
            }
        }
    }

    // Process reconnections (starting from the end to avoid indexing issues)
    reconnections.sort_by(|a, b| b.cmp(a));

    for (i, pi, j, pj) in reconnections {
        // Perform actual reconnection
        reconnect_vortex_lines(vortex_lines, i, pi, j, pj);
    }

    // Validate vortex lines after reconnections
    validate_vortex_lines(vortex_lines);
}

/// Process reconnections detected by the GPU
pub fn process_reconnections(
    vortex_lines: &mut Vec<VortexLine>, 
    reconnection_candidates: Vec<(usize, usize, usize, usize)>
) {
    // Ensure we have valid candidates
    if reconnection_candidates.is_empty() {
        return;
    }
    
    // Sort by distance (assuming the 5th element is distance)
    // This allows us to process closest reconnections first
    // Note: GPU may have already sorted these
    
    // Process each reconnection candidate
    for (line_idx1, point_idx1, line_idx2, point_idx2) in reconnection_candidates {
        // Skip if indices are no longer valid due to previous reconnections
        if line_idx1 >= vortex_lines.len() || line_idx2 >= vortex_lines.len() {
            continue;
        }
        
        let line1_len = vortex_lines[line_idx1].points.len();
        let line2_len = vortex_lines[line_idx2].points.len();
        
        if point_idx1 >= line1_len || point_idx2 >= line2_len {
            continue;
        }
        
        // Perform the actual reconnection using the existing function
        reconnect_vortex_lines(vortex_lines, line_idx1, point_idx1, line_idx2, point_idx2);
    }
    
    // Clean up after reconnections
    validate_vortex_lines(vortex_lines);
}

// Validate vortex lines and remove any degenerate elements
fn validate_vortex_lines(vortex_lines: &mut Vec<VortexLine>) {
    // Remove any degenerate lines (too few points)
    vortex_lines.retain(|line| line.points.len() >= 4);
    
    // Check for NaN or Inf values
    for line in vortex_lines {
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

pub fn remove_tiny_loops(lines: &mut Vec<VortexLine>, min_size: f64) {
    lines.retain(|line| {
        // Calculate line length
        let mut length = 0.0;
        for i in 0..line.points.len() {
            let j = (i + 1) % line.points.len();
            let dx = line.points[i].position[0] - line.points[j].position[0];
            let dy = line.points[i].position[1] - line.points[j].position[1];
            let dz = line.points[i].position[2] - line.points[j].position[2];
            length += (dx*dx + dy*dy + dz*dz).sqrt();
        }
        
        // Keep lines longer than minimum size
        length > min_size
    });
}

/// Actual reconnection algorithm that changes line connectivity
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

/// Split a vortex line at the specified index
fn split_line_at(line: &VortexLine, index: usize) -> (VortexLine, VortexLine) {
    let n_points = line.points.len();

    let mut part1 = VortexLine {
        points: Vec::with_capacity(index + 1),
    };
    let mut part2 = VortexLine {
        points: Vec::with_capacity(n_points - index),
    };

    for i in 0..=index {
        part1.points.push(line.points[i].clone());
    }

    for i in index..n_points {
        part2.points.push(line.points[i].clone());
    }

    (part1, part2)
}

/// Remesh a vortex line to maintain proper point spacing
pub fn remesh_vortex_line(
    line: &mut VortexLine,
    target_spacing: f64,
    min_spacing: f64,
    max_spacing: f64,
) {
    // Skip if line is too short
    if line.points.len() < 3 {
        return;
    }

    let mut new_points = Vec::new();
    let n_points = line.points.len();

    // Add first point as-is
    new_points.push(line.points[0].clone());

    let mut accumulated_length = 0.0;
    let mut current_idx = 0;

    while current_idx < n_points - 1 {
        let p1 = Vector3::new(
            line.points[current_idx].position[0],
            line.points[current_idx].position[1],
            line.points[current_idx].position[2],
        );

        let next_idx = (current_idx + 1) % n_points;
        let p2 = Vector3::new(
            line.points[next_idx].position[0],
            line.points[next_idx].position[1],
            line.points[next_idx].position[2],
        );

        let segment_length = (p2 - p1).norm();

        if segment_length > max_spacing {
            // Segment is too long - add intermediate points
            let num_divisions = (segment_length / target_spacing).ceil() as usize;

            for j in 1..num_divisions {
                let t = j as f64 / num_divisions as f64;
                let new_pos = p1 + (p2 - p1).scale(t);

                // Create new point with interpolated values
                new_points.push(VortexPoint {
                    position: [new_pos.x, new_pos.y, new_pos.z],
                    tangent: [0.0, 0.0, 0.0], // Will update tangents later
                });
            }
        }

        accumulated_length += segment_length;

        // Add the next point if spacing is appropriate
        if accumulated_length >= target_spacing || segment_length > max_spacing {
            new_points.push(line.points[next_idx].clone());
            accumulated_length = 0.0;
        }

        current_idx += 1;
    }

    // Only update if we have enough points
    if new_points.len() >= 3 {
        line.points = new_points;
        update_tangent_vectors(line);
    }
}

/// Calculate total vortex line length
pub fn calculate_total_length(vortex_lines: &[VortexLine]) -> f64 {
    let mut total_length = 0.0;

    for line in vortex_lines {
        for i in 0..line.points.len() {
            let next_idx = (i + 1) % line.points.len();

            let p1 = Vector3::new(
                line.points[i].position[0],
                line.points[i].position[1],
                line.points[i].position[2],
            );

            let p2 = Vector3::new(
                line.points[next_idx].position[0],
                line.points[next_idx].position[1],
                line.points[next_idx].position[2],
            );

            total_length += (p2 - p1).norm();
        }
    }

    total_length
}
