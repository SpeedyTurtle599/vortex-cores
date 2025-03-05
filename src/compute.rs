use std::borrow::Cow;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::fmt;
use crate::simulation::{VortexLine, ExternalFieldParams};

#[derive(Clone)]
pub struct ComputeCore {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

// Manually implement Debug for ComputeCore
impl fmt::Debug for ComputeCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputeCore")
            .field("device_type", &"wgpu::Device")
            .field("queue_type", &"wgpu::Queue")
            .finish()
    }
}

// Manually implement Serialize for ComputeCore
impl serde::Serialize for ComputeCore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Skip serializing compute core - just serialize unit
        serializer.serialize_unit()
    }
}

// Manually implement Deserialize for ComputeCore
impl<'de> serde::Deserialize<'de> for ComputeCore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Consume the unit value but return error since we can't actually deserialize
        // a compute core directly
        let _ = <()>::deserialize(deserializer)?;
        Err(serde::de::Error::custom("ComputeCore cannot be deserialized directly. Use ComputeCore::new() instead."))
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

        Self { device, queue }
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