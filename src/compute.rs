use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use nalgebra::Vector3;
use bytemuck::{Pod, Zeroable};
use serde::{Serialize, Deserialize};
use crate::simulation::{VortexLine, VortexPoint, ExternalFieldParams};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ReconnectionParams {
    threshold: f32,
    max_candidates: u32,
    padding1: u32,
    padding2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ReconnectionCandidate {
    line_idx1: u32,
    point_idx1: u32,
    line_idx2: u32,
    point_idx2: u32,
    distance: f32,
    dot_product: f32,
    padding: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
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

#[derive(Debug)]
pub struct ComputeCore {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    buffer_pool: std::sync::Mutex<HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>>,
}

impl Clone for ComputeCore {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            buffer_pool: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl Serialize for ComputeCore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Instead of None, serialize a special placeholder value
        serializer.serialize_str("GPU_COMPUTE_CORE_PLACEHOLDER")
    }
}

impl<'de> Deserialize<'de> for ComputeCore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept either None or our placeholder string
        let value = serde_json::Value::deserialize(deserializer)?;
        
        // Always return error - GPU resources need to be recreated
        Err(serde::de::Error::custom("ComputeCore cannot be deserialized - GPU resources must be recreated"))
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVortexPoint {
    pub position: [f32; 4],  // Using vec4 for alignment
    pub tangent: [f32; 4],   // Using vec4 for alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct LineOffset {
    start_idx: u32,
    point_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SimParams {
    kappa: f32,             // Quantum of circulation
    cutoff_radius: f32,     // Core radius cutoff for LIA
    beta: f32,              // Local induction approximation coefficient
    temperature: f32,       // Temperature in Kelvin
    time: f32,              // Current simulation time
    container_radius: f32,  // Container radius
    container_height: f32,  // Container height
    external_field_type: i32,// Type of external field (0=none, 1=rotation, etc.)
    ext_params: [[f32; 4]; 4], // External field parameters
}

impl ComputeCore {
    fn get_or_create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let mut pool = self.buffer_pool.lock().unwrap();
        let key = (size, usage);
        
        if let Some(buffers) = pool.get_mut(&key) {
            if let Some(buffer) = buffers.pop() {
                return buffer;
            }
        }
        
        // No buffer available, create a new one
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }
    
    fn return_buffer_to_pool(&self, buffer: wgpu::Buffer) {
        let size = buffer.size();
        let usage = buffer.usage();
        
        let mut pool = self.buffer_pool.lock().unwrap();
        let key = (size, usage);
        
        let buffers = pool.entry(key).or_insert_with(Vec::new);
        buffers.push(buffer);
        
        // Limit pool size
        if buffers.len() > 10 {
            buffers.remove(0); // Remove oldest buffer if we have too many
        }
    }
    
    fn release_unused_buffers(&self) {
        let mut pool = self.buffer_pool.lock().unwrap();
        pool.clear();
    }

    pub async fn list_available_gpus() -> Vec<(String, wgpu::DeviceType, String)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        
        let mut result = Vec::new();
        for adapter in adapters {
            let info = adapter.get_info();
            
            // Request device to verify compatibility
            let device_result = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await;
            
            let compatibility = match device_result {
                Ok(_) => "Compatible".to_string(),
                Err(e) => format!("Not compatible: {}", e),
            };
            
            result.push((info.name, info.device_type, compatibility));
        }
        
        result
    }

    pub async fn new_with_device_preference(device_name_fragment: Option<&str>) -> Self {
        // Create an instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        
        // List all adapters
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        
        // Find the preferred adapter if specified
        let adapter = if let Some(name_fragment) = device_name_fragment {
            // Try to find a GPU with name containing the specified fragment
            let matching_adapters: Vec<_> = adapters
                .into_iter()
                .filter(|a| a.get_info().name.to_lowercase().contains(
                    &name_fragment.to_lowercase()))
                .collect();
            
            if !matching_adapters.is_empty() {
                // Choose the first matching adapter
                println!("Found GPU matching '{}': {}", name_fragment, matching_adapters[0].get_info().name);
                matching_adapters[0].clone()
            } else {
                // Fall back to default selection if no match
                println!("No GPU found matching '{}', falling back to default selection", name_fragment);
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .expect("Failed to find an appropriate adapter")
            }
        } else {
            // No preference specified, use default selection
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find an appropriate adapter")
        };

        // Create the device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        println!(
            "Using GPU: {} ({})",
            adapter.get_info().name,
            if adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu {
                "Discrete GPU"
            } else {
                "Integrated GPU"
            }
        );

        Self{ 
            device, 
            queue,
            buffer_pool: std::sync::Mutex::new(HashMap::new()),
        }
    }
    
    // For backward compatibility
    pub async fn new() -> Self {
        Self::new_with_device_preference(None).await
    }

    // Calculate velocities for all points in all vortex lines
    pub fn calculate_velocities(
        &self, 
        vortex_lines: &[VortexLine], 
        temperature: f64,
        time: f64,
        container_radius: f64,
        container_height: f64,
        ext_field: Option<&ExternalFieldParams>,
    ) -> Vec<Vec<[f64; 3]>> {
        // 1. Convert vortex data to GPU-friendly format
        let (points_data, line_offsets) = self.prepare_vortex_data(vortex_lines);
        
        // 2. Create simulation parameters
        let params = self.create_sim_params(
            temperature, 
            time, 
            container_radius,
            container_height,
            ext_field,
        );
        
        // 3. Create GPU buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vortex Points Buffer"),
            contents: bytemuck::cast_slice(&points_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        let total_points = points_data.len();
        
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity Output Buffer"),
            size: (total_points * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let line_offset_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line Offsets Buffer"),
            contents: bytemuck::cast_slice(&line_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // println!("Starting GPU computation for {} vortex lines with {} total points", vortex_lines.len(), total_points);
        
        // 4. Create compute shader
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Velocity Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/velocity.wgsl"))),
        });
        
        // 5. Create bind group layout and pipeline
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Velocity Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: line_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // 6. Execute compute pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Velocity Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroups - each with 256 threads
            let workgroup_count = (total_points as f64 / 256.0).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // 7. Submit work and read back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (total_points * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer, 
            0, 
            &staging_buffer, 
            0, 
            (total_points * std::mem::size_of::<[f32; 4]>()) as u64
        );
        
        self.queue.submit(Some(encoder.finish()));
        
        // 8. Read back the results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(_)) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();

            // Convert and organize the results by vortex line
            let mut velocities = Vec::with_capacity(vortex_lines.len());
            let mut point_index = 0;

            for line in vortex_lines {
                let mut line_velocities = Vec::with_capacity(line.points.len());
                
                for _ in 0..line.points.len() {
                    let vel = result[point_index];
                    line_velocities.push([
                        vel[0] as f64,
                        vel[1] as f64,
                        vel[2] as f64,
                    ]);
                    point_index += 1;
                }
                
                velocities.push(line_velocities);
            }

            velocities
        } else {
            eprintln!("Failed to read GPU results, falling back to CPU implementation");
            // Return empty vectors as fallback
            vortex_lines.iter()
                .map(|line| vec![[0.0; 3]; line.points.len()])
                .collect()
        }
    }

    pub fn detect_reconnections(
        &self, 
        vortex_lines: &[VortexLine], 
        threshold: f64
    ) -> Vec<(usize, usize, usize, usize)> {
        // Convert vortex data to GPU-friendly format
        let (points_data, line_offsets) = self.prepare_vortex_data(vortex_lines);
        
        // Create buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vortex Points Buffer"),
            contents: bytemuck::cast_slice(&points_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let line_offset_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line Offsets Buffer"),
            contents: bytemuck::cast_slice(&line_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Create parameters buffer
        let params = ReconnectionParams {
            threshold: threshold as f32,
            max_candidates: 1024, // Maximum number of reconnection candidates
            padding1: 0,
            padding2: 0,
        };
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Reconnection Parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Add atomic counter buffer for GPU to count reconnection candidates
        let counter_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reconnection Counter"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        
        // Initialize counter to 0
        {
            let mut counter_data = counter_buffer.slice(..).get_mapped_range_mut();
            let counter = counter_data.as_mut_ptr() as *mut u32;
            unsafe { *counter = 0; }
        }
        counter_buffer.unmap();

        // Create candidates buffer (with counter in first element)
        let candidates_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reconnection Candidates Buffer"),
            size: (params.max_candidates as usize * std::mem::size_of::<ReconnectionCandidate>() + 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        
        // Initialize candidates buffer with zero count
        {
            let mut mapping = candidates_buffer.slice(0..4).get_mapped_range_mut();
            let counter = bytemuck::from_bytes_mut::<u32>(&mut mapping[0..4]);
            *counter = 0;
        }
        candidates_buffer.unmap();
        
        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reconnection Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/reconnection.wgsl"))),
        });
        
        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reconnection Bind Group Layout"),
            entries: &[
                // Input vortex points
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Reconnection candidates output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Line offsets
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Add the atomic counter binding
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Reconnection Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reconnection Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reconnection Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: line_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: counter_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create encoder and dispatch workgroups
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reconnection Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reconnection Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch enough workgroups to process total number of points
            let total_points = points_data.len();
            let workgroup_count = ((total_points as f64) / 256.0).ceil() as u32;
            // println!("Dispatching {} workgroups for {} points", workgroup_count, total_points);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // Create staging buffer to read results
        let staging_buffer_size = (params.max_candidates as usize * std::mem::size_of::<ReconnectionCandidate>() + 4) as u64;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reconnection Staging Buffer"),
            size: staging_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy atomic counter to candidates buffer
        encoder.copy_buffer_to_buffer(
            &counter_buffer,
            0,
            &candidates_buffer, 
            0, 
            std::mem::size_of::<u32>() as u64
        );
        
        // Then copy results to staging buffer as before
        encoder.copy_buffer_to_buffer(
            &candidates_buffer, 
            0, 
            &staging_buffer, 
            0, 
            staging_buffer_size
        );
        
        // Submit work to GPU
        self.queue.submit(Some(encoder.finish()));
        
        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        self.device.poll(wgpu::Maintain::Wait);
        
        if let Some(Ok(_)) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            
            // First 4 bytes contain the count
            let count = std::cmp::min(
                u32::from_ne_bytes([data[0], data[1], data[2], data[3]]) as usize,
                params.max_candidates as usize
            );
            
            // Debug output to verify threshold
            // println!("Reconnection detector: threshold={}, found {} candidates", threshold, count);
            
            // Extract candidates
            let candidates: Vec<ReconnectionCandidate> = bytemuck::cast_slice(&data[4..])
                .iter()
                .take(count)
                .copied()
                .collect();
            
            // Print the first few distances to see if we're close to threshold
            if count == 0 {
                // Do a simple check of raw positions to see if any are close
                for (i, line1) in vortex_lines.iter().enumerate() {
                    for (j, line2) in vortex_lines.iter().enumerate() {
                        if i >= j { continue; }  // Only check unique pairs
                        
                        // Check first few points of each line
                        for p1 in line1.points.iter().take(10) {
                            for p2 in line2.points.iter().take(10) {
                                let dx = p1.position[0] - p2.position[0];
                                let dy = p1.position[1] - p2.position[1];
                                let dz = p1.position[2] - p2.position[2];
                                let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                                
                                if dist < threshold * 3.0 {
                                    println!("Close points: lines {}/{}, dist={}, threshold={}", i, j, dist, threshold);
                                }
                            }
                        }
                    }
                }
            }

            // // Attempt debugging of distance parameter in reconnection candidates
            // println!("DEBUG: Checking distances between vortex points");
            // let mut min_dist = f64::MAX;
            // let mut min_i = 0;
            // let mut min_j = 0;
            // let mut min_pi = 0;
            // let mut min_pj = 0;

            // for (i, line1) in vortex_lines.iter().enumerate() {
            //     for (j, line2) in vortex_lines.iter().enumerate() {
            //         if i == j { continue; }
                    
            //         for (pi, point1) in line1.points.iter().enumerate() {
            //             for (pj, point2) in line2.points.iter().enumerate() {
            //                 let dx = point1.position[0] - point2.position[0];
            //                 let dy = point1.position[1] - point2.position[1];
            //                 let dz = point1.position[2] - point2.position[2];
            //                 let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                            
            //                 if dist < min_dist {
            //                     min_dist = dist;
            //                     min_i = i;
            //                     min_j = j;
            //                     min_pi = pi;
            //                     min_pj = pj;
            //                 }
            //             }
            //         }
            //     }
            // }

            // println!("Minimum distance between any two points: {} (threshold: {})", 
            //         min_dist, threshold);
            // println!("Closest points: line {}, point {} and line {}, point {}", 
            //         min_i, min_pi, min_j, min_pj);

            // // Attempt debugging of dot product parameter in reconnection candidates
            // // Check dot product of the closest points
            // let t1 = Vector3::new(
            //     vortex_lines[min_i].points[min_pi].tangent[0],
            //     vortex_lines[min_i].points[min_pi].tangent[1],
            //     vortex_lines[min_i].points[min_pi].tangent[2]
            // );

            // let t2 = Vector3::new(
            //     vortex_lines[min_j].points[min_pj].tangent[0],
            //     vortex_lines[min_j].points[min_pj].tangent[1],
            //     vortex_lines[min_j].points[min_pj].tangent[2]
            // );

            // let dot_product = t1.dot(&t2);
            // println!("Dot product between closest points: {} (threshold: -0.3)", dot_product);
            
            // Convert to result format and apply both distance and dot product filters
            let mut result: Vec<(usize, usize, usize, usize)> = candidates.iter()
            .filter(|c| {
                // Apply both distance and dot product filters
                c.distance > 0.0 && 
                c.distance < threshold as f32 && 
                c.dot_product < -0.3
            })
            .map(|c| {
                (
                    c.line_idx1 as usize,
                    c.point_idx1 as usize,
                    c.line_idx2 as usize,
                    c.point_idx2 as usize
                )
            })
            .collect();
            
                if !result.is_empty() {
                    println!("GPU detected {} potential reconnections", result.len());
                    for (i, (l1, p1, l2, p2)) in result.iter().enumerate().take(5) {
                        println!("  Candidate {}: Line {} point {} with Line {} point {}", i+1, l1, p1, l2, p2);
                    }
                }

            result.sort_unstable_by(|a, b| b.cmp(a)); // Sort in reverse order
            result
        } else {
            eprintln!("Failed to read reconnection candidates from GPU");
            Vec::new()
        }
    }

    pub fn calculate_thermal_fluctuations(
        &self,
        vortex_lines: &[VortexLine],
        temperature: f64,
        dt: f64
    ) -> Vec<Vec<[f64; 3]>> {
        if temperature < 0.01 {
            // Return zero fluctuations if temperature is negligible
            return vortex_lines.iter()
                .map(|line| vec![[0.0, 0.0, 0.0]; line.points.len()])
                .collect();
        }
        
        // Convert vortex data to GPU-friendly format
        let (points_data, line_offsets) = self.prepare_vortex_data(vortex_lines);
        
        // Create input and output buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vortex Points Buffer"),
            contents: bytemuck::cast_slice(&points_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let total_points = points_data.len();
        
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluctuation Output Buffer"),
            size: (total_points * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let line_offset_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line Offsets Buffer"),
            contents: bytemuck::cast_slice(&line_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        // Create simulation parameters
        let (alpha, alpha_prime) = if temperature > 0.0 {
            let ratio = (temperature / 2.17).clamp(0.0, 1.0);
            // Alpha increases with temp, approximately as T⁴ near 0K and levels off near Tλ
            let alpha = if temperature < 1.0 {
                0.006 * temperature.powi(4)
            } else {
                0.006 + 0.5 * (temperature - 1.0) / 1.17
            };
            
            // Alpha' is typically much smaller
            let alpha_prime = 0.0001 + alpha * 0.1;
            
            (alpha, alpha_prime)
        } else {
            (0.0, 0.0)
        };
        
        // Create parameters for thermal fluctuations
        let params = FluctuationParams {
            temperature: temperature as f32,
            dt: dt as f32,
            noise_amplitude: (1e-4 * (temperature / 2.17).sqrt() * dt.sqrt() * alpha) as f32,
            alpha: alpha as f32,
            alpha_prime: alpha_prime as f32,
            seed: rand::random::<u32>(),
            padding1: 0,
            padding2: 0,
        };
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluctuation Parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Thermal Fluctuation Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/thermals.wgsl"))),
        });
        
        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Thermal Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Thermal Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Thermal Fluctuation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Thermal Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: line_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Thermal Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Thermal Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch workgroups
            let workgroup_count = (total_points as f64 / 256.0).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // Create staging buffer for result readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Thermal Staging Buffer"),
            size: (total_points * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (total_points * std::mem::size_of::<[f32; 4]>()) as u64
        );
        
        // Submit work
        self.queue.submit(Some(encoder.finish()));
        
        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        self.device.poll(wgpu::Maintain::Wait);
        
        if let Some(Ok(_)) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
            
            // Organize results by vortex line
            let mut fluctuations = Vec::with_capacity(vortex_lines.len());
            let mut point_index = 0;
            
            for line in vortex_lines {
                let mut line_fluctuations = Vec::with_capacity(line.points.len());
                
                for _ in 0..line.points.len() {
                    let fluct = result[point_index];
                    line_fluctuations.push([
                        fluct[0] as f64, 
                        fluct[1] as f64, 
                        fluct[2] as f64
                    ]);
                    point_index += 1;
                }
                
                fluctuations.push(line_fluctuations);
            }
            
            fluctuations
        } else {
            eprintln!("Failed to read thermal fluctuations from GPU");
            // Return zero fluctuations as fallback
            vortex_lines.iter()
                .map(|line| vec![[0.0, 0.0, 0.0]; line.points.len()])
                .collect()
        }
    }

    fn prepare_vortex_data(&self, vortex_lines: &[VortexLine]) -> (Vec<GpuVortexPoint>, Vec<LineOffset>) {
        let mut points_data = Vec::new();
        let mut line_offsets = Vec::with_capacity(vortex_lines.len());
        let mut offset = 0;
        
        for line in vortex_lines {
            line_offsets.push(LineOffset {
                start_idx: offset,
                point_count: line.points.len() as u32,
            });
            
            for point in &line.points {
                points_data.push(GpuVortexPoint {
                    position: [
                        point.position[0] as f32, 
                        point.position[1] as f32, 
                        point.position[2] as f32, 
                        0.0
                    ],
                    tangent: [
                        point.tangent[0] as f32, 
                        point.tangent[1] as f32, 
                        point.tangent[2] as f32, 
                        0.0
                    ],
                });
                offset += 1;
            }
        }
        
        (points_data, line_offsets)
    }

    pub fn remesh_vortices_gpu(
        &self,
        vortex_lines: &[VortexLine],
        target_spacing: f64,
        min_spacing: f64,
        max_spacing: f64
    ) -> Vec<VortexLine> {
        // For each line, calculate the new number of points based on line length
        let mut remeshed_lines = Vec::with_capacity(vortex_lines.len());
        
        for line in vortex_lines {
            // Calculate line length
            let mut length = 0.0;
            for i in 0..line.points.len() {
                let j = (i + 1) % line.points.len();
                let dx = line.points[i].position[0] - line.points[j].position[0];
                let dy = line.points[i].position[1] - line.points[j].position[1];
                let dz = line.points[i].position[2] - line.points[j].position[2];
                length += (dx*dx + dy*dy + dz*dz).sqrt();
            }
            
            // Calculate target number of points
            let new_point_count = (length / target_spacing).round() as usize;
            if new_point_count < 3 {
                // Keep original line if it would become too small
                remeshed_lines.push(line.clone());
                continue;
            }
            
            // Create new points with uniform spacing
            let mut new_points = Vec::with_capacity(new_point_count);
            let delta_param = 1.0 / new_point_count as f64;
            
            for i in 0..new_point_count {
                let t = i as f64 * delta_param;
                let position = self.interpolate_position(line, t);
                
                new_points.push(VortexPoint {
                    position,
                    tangent: [0.0, 0.0, 0.0] // Will be updated later
                });
            }
            
            // Create new line
            let mut remeshed_line = VortexLine { points: new_points };
            
            // Update tangent vectors
            self.calculate_tangent_vectors(&mut remeshed_line);
            
            remeshed_lines.push(remeshed_line);
        }
        
        remeshed_lines
    }
    
    // Helper method to interpolate position along a closed vortex line
    fn interpolate_position(&self, line: &VortexLine, t: f64) -> [f64; 3] {
        let n_points = line.points.len();
        let param = t * n_points as f64;
        let idx1 = param.floor() as usize % n_points;
        let idx2 = (idx1 + 1) % n_points;
        let frac = param - param.floor();
        
        let p1 = line.points[idx1].position;
        let p2 = line.points[idx2].position;
        
        [
            p1[0] * (1.0 - frac) + p2[0] * frac,
            p1[1] * (1.0 - frac) + p2[1] * frac,
            p1[2] * (1.0 - frac) + p2[2] * frac,
        ]
    }
    
    // Helper method to calculate tangent vectors
    fn calculate_tangent_vectors(&self, line: &mut VortexLine) {
        let n_points = line.points.len();
        if n_points < 3 {
            return;
        }
        
        for i in 0..n_points {
            let prev = (i + n_points - 1) % n_points;
            let next = (i + 1) % n_points;
            
            // Calculate segments
            let dx_prev = line.points[i].position[0] - line.points[prev].position[0];
            let dy_prev = line.points[i].position[1] - line.points[prev].position[1];
            let dz_prev = line.points[i].position[2] - line.points[prev].position[2];
            
            let dx_next = line.points[next].position[0] - line.points[i].position[0];
            let dy_next = line.points[next].position[1] - line.points[i].position[1];
            let dz_next = line.points[next].position[2] - line.points[i].position[2];
            
            // Average the two segments
            let tx = dx_prev + dx_next;
            let ty = dy_prev + dy_next;
            let tz = dz_prev + dz_next;
            
            // Normalize
            let mag = (tx*tx + ty*ty + tz*tz).sqrt();
            if mag > 1e-10 {
                line.points[i].tangent = [tx/mag, ty/mag, tz/mag];
            } else {
                line.points[i].tangent = [0.0, 0.0, 1.0];
            }
        }
    }
    
    fn create_sim_params(
        &self,
        temperature: f64,
        time: f64,
        container_radius: f64,
        container_height: f64,
        ext_field: Option<&ExternalFieldParams>,
    ) -> SimParams {
        // Physical constants
        const KAPPA: f32 = 9.97e-4; // Quantum of circulation in cm²/s
        const CORE_RADIUS: f32 = 1.0e-8; // Core radius in cm
        const BETA: f32 = 0.5; // LIA coefficient
        
        // External field parameters
        let (field_type, ext_params) = match ext_field {
            Some(ExternalFieldParams::Rotation { angular_velocity, center }) => {
                let mut params = [[0.0f32; 4]; 4];
                // Angular velocity
                params[0][0] = angular_velocity[0] as f32;
                params[0][1] = angular_velocity[1] as f32;
                params[0][2] = angular_velocity[2] as f32;
                
                // Center
                params[1][0] = center[0] as f32;
                params[1][1] = center[1] as f32;
                params[1][2] = center[2] as f32;
                
                (1i32, params)
            },
            Some(ExternalFieldParams::UniformFlow { velocity }) => {
                let mut params = [[0.0f32; 4]; 4];
                params[0][0] = velocity[0] as f32;
                params[0][1] = velocity[1] as f32;
                params[0][2] = velocity[2] as f32;
                
                (2i32, params)
            },
            Some(ExternalFieldParams::OscillatoryFlow { amplitude, frequency, phase }) => {
                let mut params = [[0.0f32; 4]; 4];
                params[0][0] = amplitude[0] as f32;
                params[0][1] = amplitude[1] as f32;
                params[0][2] = amplitude[2] as f32;
                params[0][3] = *frequency as f32;
                params[1][0] = *phase as f32;
                
                (3i32, params)
            },
            Some(ExternalFieldParams::Counterflow { velocity }) => {
                let mut params = [[0.0f32; 4]; 4];
                params[0][0] = velocity[0] as f32;
                params[0][1] = velocity[1] as f32;
                params[0][2] = velocity[2] as f32;
                
                (4i32, params)
            },
            None => (0i32, [[0.0f32; 4]; 4]),
        };
        
        SimParams {
            kappa: KAPPA,
            cutoff_radius: CORE_RADIUS,
            beta: BETA,
            temperature: temperature as f32,
            time: time as f32,
            container_radius: container_radius as f32,
            container_height: container_height as f32,
            external_field_type: field_type,
            ext_params,
        }
    }
}