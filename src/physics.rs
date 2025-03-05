use crate::simulation::{VortexLine, VortexPoint};
use std::f64::consts::PI;

// Physical constants for superfluid helium
const QUANTUM_CIRCULATION: f64 = 9.97e-4; // cm²/s (κ = h/m)
const CORE_RADIUS: f64 = 1.0e-8; // cm, approximate vortex core radius
const MUTUAL_FRICTION_ALPHA: f64 = 0.1; // Normal fluid coupling coefficient
const MUTUAL_FRICTION_ALPHA_PRIME: f64 = 0.0; // Transverse coupling coefficient

// LIA parameter: β ≈ ln(L/a) where L is typical filament length and a is core radius
// Typically β ≈ 10-12 for superfluid helium
const LIA_BETA: f64 = 10.0;

// Flag to enable hybrid LIA+Biot-Savart model
const USE_HYBRID_MODEL: bool = true;

/// MARK: Biot-Savart
/// Calculate Biot-Savart law for vortex filaments with de-singularized core
fn biot_savart(source: &[f64; 3], target: &[f64; 3], tangent: &[f64; 3]) -> [f64; 3] {
    // Vector from source to target
    let r = [
        target[0] - source[0],
        target[1] - source[1],
        target[2] - source[2],
    ];
    
    // Distance squared
    let r_mag_squared = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    
    // De-singularized denominator (Rosenhead cut-off)
    let denominator = (r_mag_squared + CORE_RADIUS*CORE_RADIUS).powf(1.5);
    
    if denominator < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    
    // Cross product: tangent × r
    let cross = [
        tangent[1] * r[2] - tangent[2] * r[1],
        tangent[2] * r[0] - tangent[0] * r[2],
        tangent[0] * r[1] - tangent[1] * r[0],
    ];
    
    // Scale by 1/(4π(r² + a²)^(3/2))
    let scale = 1.0 / (4.0 * PI * denominator);
    
    [
        cross[0] * scale,
        cross[1] * scale,
        cross[2] * scale,
    ]
}

/// MARK: Calculate Velocity
/// Calculate the velocity at a point due to all vortex segments
fn calculate_velocity(
    point: &[f64; 3],
    vortex_lines: &[VortexLine],
    exclude_line: Option<usize>,
    exclude_point: Option<usize>,
) -> [f64; 3] {
    let mut velocity = [0.0, 0.0, 0.0];
    
    // For each vortex line
    for (line_idx, line) in vortex_lines.iter().enumerate() {
        // Skip if this line is excluded
        if exclude_line == Some(line_idx) {
            continue;
        }
        
        // For each segment in the line
        for (point_idx, vortex_point) in line.points.iter().enumerate() {
            // Skip if this point is excluded
            if exclude_line == Some(line_idx) && exclude_point == Some(point_idx) {
                continue;
            }
            
            // Get contribution from this segment via Biot-Savart
            let segment_velocity = biot_savart(&vortex_point.position, point, &vortex_point.tangent);
            
            // Add to total velocity (scaled by quantum circulation)
            velocity[0] += segment_velocity[0] * QUANTUM_CIRCULATION;
            velocity[1] += segment_velocity[1] * QUANTUM_CIRCULATION;
            velocity[2] += segment_velocity[2] * QUANTUM_CIRCULATION;
        }
    }
    
    velocity
}

/// MARK: Mutual Friction
/// Apply mutual friction effects based on temperature
fn apply_mutual_friction(
    velocity: [f64; 3], 
    tangent: &[f64; 3],
    temperature: f64
) -> [f64; 3] {
    // Skip if at absolute zero
    if temperature < 0.001 {
        return velocity;
    }
    
    // Temperature-dependent mutual friction coefficients
    // This is a simple model - in reality, the relationship is more complex
    let alpha = MUTUAL_FRICTION_ALPHA * temperature / 2.17; // normalized by Tc
    let alpha_prime = MUTUAL_FRICTION_ALPHA_PRIME * temperature / 2.17;
    
    // Calculate mutual friction components
    
    // s' × (s' × v) component
    let s_cross_v = [
        tangent[1] * velocity[2] - tangent[2] * velocity[1],
        tangent[2] * velocity[0] - tangent[0] * velocity[2],
        tangent[0] * velocity[1] - tangent[1] * velocity[0],
    ];
    
    let s_cross_s_cross_v = [
        tangent[1] * s_cross_v[2] - tangent[2] * s_cross_v[1],
        tangent[2] * s_cross_v[0] - tangent[0] * s_cross_v[2],
        tangent[0] * s_cross_v[1] - tangent[1] * s_cross_v[0],
    ];
    
    // s' × v component
    
    // Modified velocity with mutual friction
    [
        velocity[0] - alpha * s_cross_s_cross_v[0] + alpha_prime * s_cross_v[0],
        velocity[1] - alpha * s_cross_s_cross_v[1] + alpha_prime * s_cross_v[1],
        velocity[2] - alpha * s_cross_s_cross_v[2] + alpha_prime * s_cross_v[2],
    ]
}

/// MARK: LIA Velocity
/// Calculate local induction approximation (LIA) for a vortex filament
pub fn calculate_lia_velocity(
    points: &[VortexPoint],
    idx: usize,
    beta: f64
) -> [f64; 3] {
    // Get neighboring points
    let num_points = points.len();
    let prev_idx = (idx + num_points - 1) % num_points;
    let next_idx = (idx + 1) % num_points;
    
    let r_prev = &points[prev_idx].position;
    let r = &points[idx].position;
    let r_next = &points[next_idx].position;
    
    // Calculate segments
    let segment1 = [
        r[0] - r_prev[0],
        r[1] - r_prev[1],
        r[2] - r_prev[2],
    ];
    
    let segment2 = [
        r_next[0] - r[0],
        r_next[1] - r[1],
        r_next[2] - r[2],
    ];
    
    // Calculate lengths
    let len1 = (segment1[0]*segment1[0] + segment1[1]*segment1[1] + segment1[2]*segment1[2]).sqrt();
    let len2 = (segment2[0]*segment2[0] + segment2[1]*segment2[1] + segment2[2]*segment2[2]).sqrt();
    
    if len1 < 1e-10 || len2 < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    
    // Calculate unit tangent vectors
    let t1 = [segment1[0]/len1, segment1[1]/len1, segment1[2]/len1];
    let t2 = [segment2[0]/len2, segment2[1]/len2, segment2[2]/len2];
    
    // Calculate binormal (t1 × t2)
    let binormal = [
        t1[1]*t2[2] - t1[2]*t2[1],
        t1[2]*t2[0] - t1[0]*t2[2],
        t1[0]*t2[1] - t1[1]*t2[0],
    ];
    
    // Calculate curvature vector magnitude
    let bin_mag = (binormal[0]*binormal[0] + binormal[1]*binormal[1] + binormal[2]*binormal[2]).sqrt();
    
    // If binormal is too small, segments are nearly parallel - minimal LIA contribution
    if bin_mag < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    
    // Calculate curvature (this is an approximation)
    let curvature = bin_mag / (1.0 + t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2]);
    
    // Average tangent vector
    let tangent = [
        (t1[0] + t2[0]) / 2.0,
        (t1[1] + t2[1]) / 2.0,
        (t1[2] + t2[2]) / 2.0,
    ];
    
    // Normalize the binormal
    let binormal_normalized = [
        binormal[0] / bin_mag,
        binormal[1] / bin_mag,
        binormal[2] / bin_mag,
    ];
    
    // Calculate LIA velocity: v = β(κ × t)
    // β is logarithmic factor that includes circulation and core size effects
    [
        beta * curvature * binormal_normalized[0] * QUANTUM_CIRCULATION,
        beta * curvature * binormal_normalized[1] * QUANTUM_CIRCULATION,
        beta * curvature * binormal_normalized[2] * QUANTUM_CIRCULATION,
    ]
}

/// Evolve the vortex network forward in time
pub fn evolve_vortex_network(vortex_lines: &mut [VortexLine], dt: f64, temperature: f64) {
    // For each vortex line
    for line_idx in 0..vortex_lines.len() {
        let mut new_positions = Vec::with_capacity(vortex_lines[line_idx].points.len());
        let line_points = &vortex_lines[line_idx].points;
        
        // For each point in the line
        for point_idx in 0..line_points.len() {
            let point = &line_points[point_idx];
            let mut velocity = [0.0, 0.0, 0.0];
            
            if USE_HYBRID_MODEL {
                // LIA for self-induced velocity (much faster)
                let lia_velocity = calculate_lia_velocity(
                    &line_points, 
                    point_idx,
                    LIA_BETA
                );
                
                velocity[0] += lia_velocity[0];
                velocity[1] += lia_velocity[1];
                velocity[2] += lia_velocity[2];
                
                // Biot-Savart only for interactions with other vortex lines
                // (not for self-interaction)
                for other_idx in 0..vortex_lines.len() {
                    if other_idx == line_idx {
                        continue; // Skip self-line (already handled by LIA)
                    }
                    
                    // Calculate interaction with other lines using Biot-Savart
                    for vortex_point in &vortex_lines[other_idx].points {
                        let segment_velocity = biot_savart(
                            &vortex_point.position, 
                            &point.position, 
                            &vortex_point.tangent
                        );
                        
                        velocity[0] += segment_velocity[0] * QUANTUM_CIRCULATION;
                        velocity[1] += segment_velocity[1] * QUANTUM_CIRCULATION;
                        velocity[2] += segment_velocity[2] * QUANTUM_CIRCULATION;
                    }
                }
            } else {
                // Full Biot-Savart calculation (slower but more accurate)
                velocity = calculate_velocity(
                    &point.position,
                    vortex_lines,
                    Some(line_idx),
                    Some(point_idx)
                );
            }
            
            // Apply mutual friction effects based on temperature
            let modified_velocity = apply_mutual_friction(
                velocity, 
                &point.tangent,
                temperature
            );
            
            // Update position using semi-implicit Euler scheme
            let new_pos = [
                point.position[0] + modified_velocity[0] * dt,
                point.position[1] + modified_velocity[1] * dt,
                point.position[2] + modified_velocity[2] * dt,
            ];
            
            new_positions.push(new_pos);
        }
        
        // Apply new positions and recalculate tangents
        for (idx, new_pos) in new_positions.iter().enumerate() {
            vortex_lines[line_idx].points[idx].position = *new_pos;
        }
        
        // Update all tangents after positions have been updated
        // This ensures consistency between positions and tangents
        for idx in 0..vortex_lines[line_idx].points.len() {
            let num_points = vortex_lines[line_idx].points.len();
            let next_idx = (idx + 1) % num_points;
            let prev_idx = (idx + num_points - 1) % num_points;
            
            let p_next = &vortex_lines[line_idx].points[next_idx].position;
            let p_prev = &vortex_lines[line_idx].points[prev_idx].position;
            
            // Calculate tangent using central difference
            let dx = p_next[0] - p_prev[0];
            let dy = p_next[1] - p_prev[1];
            let dz = p_next[2] - p_prev[2];
            
            let mag = (dx*dx + dy*dy + dz*dz).sqrt();
            if mag > 1e-10 {
                vortex_lines[line_idx].points[idx].tangent = [dx/mag, dy/mag, dz/mag];
            }
        }
    }
}

/// Remesh a vortex line to maintain proper point spacing
pub fn remesh_vortex_line(line: &mut VortexLine, target_spacing: f64, min_spacing: f64, max_spacing: f64) {
    // Skip if line is too short
    if line.points.len() < 3 {
        return;
    }
    
    let mut new_points = Vec::new();
    let num_points = line.points.len();
    
    // Add first point as-is
    new_points.push(line.points[0].clone());
    
    let mut accumulated_length = 0.0;
    
    // Process all segments
    for i in 0..num_points {
        let p1 = &line.points[i].position;
        let p2 = &line.points[(i + 1) % num_points].position;
        
        // Calculate segment length
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let dz = p2[2] - p1[2];
        let segment_length = (dx*dx + dy*dy + dz*dz).sqrt();
        
        accumulated_length += segment_length;
        
        // If we've accumulated enough length, add a new point
        if accumulated_length >= target_spacing {
            // Add the next point
            new_points.push(line.points[(i + 1) % num_points].clone());
            accumulated_length = 0.0;
        } else if segment_length > max_spacing {
            // If segment is too long, add intermediate points
            let num_divisions = (segment_length / target_spacing).ceil() as usize;
            
            for j in 1..num_divisions {
                let t = j as f64 / num_divisions as f64;
                
                // Interpolate position
                let pos = [
                    p1[0] + t * dx,
                    p1[1] + t * dy,
                    p1[2] + t * dz,
                ];
                
                // Create interpolated tangent
                let t1 = &line.points[i].tangent;
                let t2 = &line.points[(i + 1) % num_points].tangent;
                
                let tangent = [
                    t1[0] + t * (t2[0] - t1[0]),
                    t1[1] + t * (t2[1] - t1[1]),
                    t1[2] + t * (t2[2] - t1[2]),
                ];
                
                // Normalize tangent
                let mag = (tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]).sqrt();
                let normalized_tangent = if mag > 1e-10 {
                    [tangent[0]/mag, tangent[1]/mag, tangent[2]/mag]
                } else {
                    [0.0, 0.0, 1.0] // Default if degenerate
                };
                
                new_points.push(VortexPoint {
                    position: pos,
                    tangent: normalized_tangent,
                });
            }
            
            new_points.push(line.points[(i + 1) % num_points].clone());
            accumulated_length = 0.0;
        }
    }
    
    // Only update if we have enough points
    if new_points.len() >= 3 {
        line.points = new_points;
    }
}

/// Check for and handle reconnections between vortex lines
pub fn handle_reconnections(vortex_lines: &mut Vec<VortexLine>, reconnection_threshold: f64) {
    // Iterate over all pairs of lines
    let num_lines = vortex_lines.len();
    
    // Vector to track reconnections to be performed
    let mut reconnections = Vec::new();
    
    // First pass: identify reconnection points
    for i in 0..num_lines {
        for j in i+1..num_lines {
            // Check for close approaches between line i and line j
            for (pi, point_i) in vortex_lines[i].points.iter().enumerate() {
                for (pj, point_j) in vortex_lines[j].points.iter().enumerate() {
                    // Calculate distance between points
                    let dx = point_i.position[0] - point_j.position[0];
                    let dy = point_i.position[1] - point_j.position[1];
                    let dz = point_i.position[2] - point_j.position[2];
                    let dist_squared = dx*dx + dy*dy + dz*dz;
                    
                    // If points are closer than threshold, mark for reconnection
                    if dist_squared < reconnection_threshold * reconnection_threshold {
                        reconnections.push((i, pi, j, pj));
                    }
                }
            }
        }
    }
    
    // Second pass: perform reconnections (simplified model)
    for (i, pi, j, pj) in reconnections {
        // This is a simplified reconnection model
        // In reality, we would need to check for appropriate conditions
        // like anti-parallel vortex segments
        
        // For a full implementation, we would:
        // 1. Split both lines at reconnection points
        // 2. Reconnect the segments in a different configuration
        // 3. Re-compute tangent vectors at and around the reconnection points
        
        // For now, we just place a marker for demonstration
        if let Some(line_i) = vortex_lines.get_mut(i) {
            if let Some(point) = line_i.points.get_mut(pi) {
                // Mark reconnection (just for visualization)
                // In a full implementation, we would actually reconnect the lines
                point.tangent = [0.0, 0.0, 1.0]; // Just a marker
            }
        }
    }
}