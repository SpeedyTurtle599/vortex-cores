@group(0) @binding(0) var<storage, read> all_points: array<VortexPoint>;
@group(0) @binding(1) var<storage, read_write> candidates: array<ReconnectionCandidate>;
@group(0) @binding(2) var<storage, read> line_offsets: array<LineOffset>;
@group(0) @binding(3) var<uniform> params: ReconnectionParams;
@group(0) @binding(4) var<storage, read_write> candidate_counter: atomic<u32>;


struct VortexPoint {
    position: vec4<f32>,
    tangent: vec4<f32>,
}

struct LineOffset {
    start_idx: u32,
    point_count: u32,
}

struct ReconnectionCandidate {
    line_idx1: u32,
    point_idx1: u32,
    line_idx2: u32,
    point_idx2: u32,
    distance: f32,
    dot_product: f32,
    padding: vec3<f32>,
}

struct ReconnectionParams {
    threshold: f32,
    max_candidates: u32,
    padding1: u32,
    padding2: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // First thread initializes counter to 0
    if (global_id.x == 0u) {
        atomicStore(&candidate_counter, 0u);
    }
    
    // Add a barrier to ensure all threads see the initialized counter
    workgroupBarrier();
    
    let point_idx = global_id.x;
    
    // Skip if out of bounds
    if (point_idx >= arrayLength(&all_points)) {
        return;
    }
    
    // Find which line this point belongs to
    var line_idx1: u32 = 0u;
    var local_idx1: u32 = 0u;
    var found_line = false;
    
    for (var i = 0u; i < arrayLength(&line_offsets); i++) {
        let start = line_offsets[i].start_idx;
        let count = line_offsets[i].point_count;
        
        if (point_idx >= start && point_idx < start + count) {
            line_idx1 = i;
            local_idx1 = point_idx - start;
            found_line = true;
            break;
        }
    }
    
    if (!found_line) {
        return;
    }
    
    // Get current point data
    let current_point = all_points[point_idx];
    let pos1 = current_point.position.xyz;
    let tangent1 = current_point.tangent.xyz;
    
    // Check all points on other lines
    for (var j = line_idx1 + 1u; j < arrayLength(&line_offsets); j++) {
        let line_offset2 = line_offsets[j];
        
        for (var k = 0u; k < line_offset2.point_count; k++) {
            let point_idx2 = line_offset2.start_idx + k;
            let pos2 = all_points[point_idx2].position.xyz;
            
            // Calculate distance
            let diff = pos1 - pos2;
            let dist = length(diff);
            
            // Skip if too far
            if (dist > params.threshold) {
                continue;
            }
            
            // Get tangent vectors for both points
            let tangent2 = all_points[point_idx2].tangent.xyz;
            
            // Check if tangents are anti-parallel (dot product negative)
            let dot_product = dot(tangent1, tangent2);
            
            if (dot_product < -0.3) { // Same threshold as in CPU code
                // Found a potential reconnection
                let count = atomicAdd(&candidate_counter, 1u);
                
                // Make sure we don't exceed buffer size
                if (count < params.max_candidates) {
                    candidates[count] = ReconnectionCandidate(
                        line_idx1, 
                        local_idx1,
                        j, 
                        k,
                        dist, 
                        dot_product,
                        vec3<f32>(0.0)
                    );
                }
            }
        }
    }
}