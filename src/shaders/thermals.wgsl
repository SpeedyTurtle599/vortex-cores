@group(0) @binding(0) var<storage, read> all_points: array<VortexPoint>;
@group(0) @binding(1) var<storage, read_write> fluctuations: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> line_offsets: array<LineOffset>;
@group(0) @binding(3) var<uniform> params: FluctuationParams;

struct VortexPoint {
    position: vec4<f32>,
    tangent: vec4<f32>,
}

struct LineOffset {
    start_idx: u32,
    point_count: u32,
}

struct FluctuationParams {
    temperature: f32,
    dt: f32,
    noise_amplitude: f32,
    alpha: f32,
    alpha_prime: f32,
    seed: u32,
    padding1: u32,
    padding2: u32,
}

// Random number generation based on pcg32
fn pcg32(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate random float between -1 and 1
fn random_float(seed: u32, offset: u32) -> f32 {
    let rand_int = pcg32(seed + offset);
    return (f32(rand_int) / 2147483647.0) * 2.0 - 1.0;
}

// Generate random vec3 with components between -1 and 1
fn random_vec3(seed: u32, id: u32) -> vec3<f32> {
    return vec3<f32>(
        random_float(seed, id * 3u),
        random_float(seed, id * 3u + 1u),
        random_float(seed, id * 3u + 2u)
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;
    
    // Check if this thread should process a point
    if (point_idx >= arrayLength(&fluctuations)) {
        return;
    }
    
    // Get current point data
    let current_point = all_points[point_idx];
    let tangent = normalize(current_point.tangent.xyz);
    
    // Find which line and local index this point belongs to
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
        fluctuations[point_idx] = vec4<f32>(0.0);
        return;
    }
    
    // Generate random vector
    var rand_vec = random_vec3(params.seed, point_idx);
    
    // Project random vector to be perpendicular to tangent
    let dot_product = dot(rand_vec, tangent);
    let perpendicular = rand_vec - tangent * dot_product;
    
    // Normalize and scale by noise amplitude
    let noise = normalize(perpendicular) * params.noise_amplitude;
    
    // Store result in output buffer
    fluctuations[point_idx] = vec4<f32>(noise, 0.0);
}