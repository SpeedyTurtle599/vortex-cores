struct VortexPoint {
    position: vec4<f32>, // Using vec4 for alignment (xyz: position, w: unused)
    tangent: vec4<f32>,  // Using vec4 for alignment (xyz: tangent, w: unused)
};

struct LineOffset {
    start_idx: u32,
    point_count: u32,
};

struct SimParams {
    kappa: f32,              // Quantum of circulation constant
    cutoff_radius: f32,      // Core radius cutoff for LIA
    beta: f32,               // Local induction approximation coefficient
    temperature: f32,        // Temperature in Kelvin
    time: f32,               // Current simulation time
    container_radius: f32,   // Container radius
    container_height: f32,   // Container height
    external_field_type: i32,  // External field type (0=none, 1=rotation, 2=uniform, 3=oscillatory, 4=counterflow)
    ext_params: array<vec4<f32>, 4>, // External field parameters
};

@group(0) @binding(0) var<storage, read> all_points: array<VortexPoint>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> line_offsets: array<LineOffset>;
@group(0) @binding(3) var<uniform> params: SimParams;

// Constants
const PI: f32 = 3.14159265359;
const WORKGROUP_SIZE: u32 = 256;

// Calculate velocity contribution from a line segment using Biot-Savart law
fn segment_velocity(pos: vec3<f32>, s1: vec3<f32>, s2: vec3<f32>) -> vec3<f32> {
    let r1 = pos - s1;
    let r2 = pos - s2;
    
    let r1_len = length(r1);
    let r2_len = length(r2);
    
    // Avoid singularities
    if (r1_len < params.cutoff_radius || r2_len < params.cutoff_radius) {
        return vec3<f32>(0.0);
    }
    
    let r1_hat = r1 / r1_len;
    let r2_hat = r2 / r2_len;
    
    // Line segment
    let seg = s2 - s1;
    let seg_len = length(seg);
    
    if (seg_len < 0.000001) {
        return vec3<f32>(0.0);
    }
    
    // Cross product
    let cross_prod = cross(r1, r2);
    let cross_len = length(cross_prod);
    
    // Avoid numerical issues
    if (cross_len < 0.000001) {
        return vec3<f32>(0.0);
    }
    
    // Biot-Savart formula
    return (params.kappa / (4.0 * PI)) * (cross_prod / (cross_len * cross_len)) * dot(seg, r1_hat - r2_hat);
}

// Calculate local induction approximation (LIA) velocity
fn calculate_lia(tangent: vec3<f32>, curvature: f32) -> vec3<f32> {
    // We need a vector perpendicular to the tangent
    var perp_vec: vec3<f32>; // initialized without value
    
    // Find a suitable perpendicular vector
    if (abs(tangent.x) < 0.9) {
        perp_vec = normalize(cross(tangent, vec3<f32>(1.0, 0.0, 0.0)));
    } else {
        perp_vec = normalize(cross(tangent, vec3<f32>(0.0, 1.0, 0.0)));
    }
    
    // Create binormal vector
    let binormal = normalize(cross(tangent, perp_vec));
    
    // LIA velocity is perpendicular to both tangent and binormal
    let normal = cross(binormal, tangent);
    
    // LIA coefficient depends on temperature through mutual friction
    let temp_factor = 1.0 - 0.1 * (params.temperature / 2.17); // Simple scaling with temperature
    
    // LIA formula: v = κ β (log(L/a)) (s' × s'') = κ β (log(L/a)) κ n
    // where β is a constant, log(L/a) is another constant related to the ratio of 
    // system size to core size, κ is the vortex curvature, and n is the normal vector
    return params.kappa * params.beta * temp_factor * curvature * normal;
}

// Calculate external field velocity at a point
fn external_field_velocity(pos: vec3<f32>, time: f32) -> vec3<f32> {
    switch(params.external_field_type) {
        // No external field
        case 0: {
            return vec3<f32>(0.0);
        }
        // Rotation
        case 1: {
            let omega = params.ext_params[0].xyz; // Angular velocity
            let center = params.ext_params[1].xyz; // Center of rotation
            let r = pos - center;
            return cross(omega, r);
        }
        // Uniform flow
        case 2: {
            return params.ext_params[0].xyz; // Velocity vector
        }
        // Oscillatory flow
        case 3: {
            let amplitude = params.ext_params[0].xyz;
            let frequency = params.ext_params[0].w;
            let phase = params.ext_params[1].x;
            return amplitude * sin(2.0 * PI * frequency * time + phase);
        }
        // Counterflow
        case 4: {
            return params.ext_params[0].xyz; // Velocity vector
        }
        // Default case
        default: {
            return vec3<f32>(0.0);
        }
    }
}

// Calculate curvature at a point using neighboring points
fn calculate_curvature(prev_pos: vec3<f32>, pos: vec3<f32>, next_pos: vec3<f32>) -> f32 {
    let seg1 = normalize(pos - prev_pos);
    let seg2 = normalize(next_pos - pos);
    
    // Calculate binormal
    let binormal = cross(seg1, seg2);
    let bin_mag = length(binormal);
    
    // Calculate curvature approximation
    let dot_product = dot(seg1, seg2);
    return bin_mag / (1.0 + dot_product);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;
    
    // Check if this thread should process a point
    if (point_idx >= arrayLength(&velocities)) {
        return;
    }
    
    // Get current point data
    let current_point = all_points[point_idx];
    let pos = current_point.position.xyz;
    let tangent = current_point.tangent.xyz;
    
    // Find which line this point belongs to and its local index
    var line_idx: u32 = 0u;
    var local_idx: u32 = 0u;
    var found_line = false;
    
    for (var i = 0u; i < arrayLength(&line_offsets); i++) {
        let start = line_offsets[i].start_idx;
        let count = line_offsets[i].point_count;
        if (point_idx >= start && point_idx < start + count) {
            line_idx = i;
            local_idx = point_idx - start;
            found_line = true;
            break;
        }
    }
    
    if (!found_line) {
        // Should never happen if data is correct
        velocities[point_idx] = vec4<f32>(0.0);
        return;
    }
    
    // Get the line offset information
    let line_offset = line_offsets[line_idx];
    let start_idx = line_offset.start_idx;
    let point_count = line_offset.point_count;
    
    // Calculate curvature using neighboring points
    var curvature: f32 = 0.1; // Default value
    
    if (point_count > 2u) {
        let prev_idx = start_idx + ((local_idx + point_count - 1u) % point_count);
        let next_idx = start_idx + ((local_idx + 1u) % point_count);
        
        let prev_pos = all_points[prev_idx].position.xyz;
        let next_pos = all_points[next_idx].position.xyz;
        
        curvature = calculate_curvature(prev_pos, pos, next_pos);
    }
    
    // 1. Calculate LIA velocity contribution
    var velocity = calculate_lia(tangent, curvature);
    
    // 2. Calculate non-local contributions from other vortex segments
    for (var i = 0u; i < arrayLength(&line_offsets); i++) {
        let other_start = line_offsets[i].start_idx;
        let other_count = line_offsets[i].point_count;
        
        // Skip points too close to current point
        if (i == line_idx && other_count < 20u) {
            continue; // Skip small loops completely (LIA handles them)
        }
        
        // Sample segments at a coarser resolution for performance
        let step = max(1u, other_count / 20u);
        
        for (var j = 0u; j < other_count - 1u; j += step) {
            let idx1 = other_start + j;
            let idx2 = other_start + j + 1u;
            
            // Skip segments too close to current point
            if (i == line_idx && abs(i32(local_idx) - i32(j)) < 5) {
                continue;
            }
            
            let s1 = all_points[idx1].position.xyz;
            let s2 = all_points[idx2].position.xyz;
            
            velocity += segment_velocity(pos, s1, s2);
        }
    }
    
    // 3. Add external field velocity
    velocity += external_field_velocity(pos, params.time);
    
    // 4. Apply mutual friction if temperature > 0
    if (params.temperature > 0.01) {
        // Calculate mutual friction coefficients
        let temp_ratio = min(params.temperature / 2.17, 1.0);
        let alpha = 0.1 * temp_ratio; 
        let alpha_prime = 0.05 * temp_ratio;
        
        // Calculate perpendicular component of velocity
        let v_parallel = tangent * dot(velocity, tangent);
        let v_perp = velocity - v_parallel;
        
        // Apply friction - reduce velocity
        let friction = alpha * cross(tangent, cross(tangent, velocity)) + 
                      alpha_prime * cross(tangent, velocity);
        
        velocity -= friction;
    }
    
    // Store the result
    velocities[point_idx] = vec4<f32>(velocity, 0.0);
}