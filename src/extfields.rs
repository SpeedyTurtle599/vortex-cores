use nalgebra::Vector3;

/// Represents different types of external flow fields
pub enum ExternalField {
    /// No external flow
    None,
    
    /// Uniform flow field in a specific direction
    UniformFlow {
        velocity: Vector3<f64>,
    },
    
    /// Solid body rotation around a specific axis
    Rotation {
        angular_velocity: Vector3<f64>,
        center: Vector3<f64>,
    },
    
    /// Oscillatory flow field
    OscillatoryFlow {
        amplitude: Vector3<f64>,
        frequency: f64,
        phase: f64,
    },
    
    /// Poiseuille flow in a pipe
    PoiseuilleFlow {
        max_velocity: f64,
        axis: Vector3<f64>,
        radius: f64,
    },
    
    /// Counterflow between normal and superfluid components
    Counterflow {
        velocity: Vector3<f64>,
    },
}

impl ExternalField {
    /// Calculate velocity at a given point and time
    pub fn velocity_at(&self, position: &Vector3<f64>, time: f64) -> Vector3<f64> {
        match self {
            ExternalField::None => Vector3::zeros(),
            
            ExternalField::UniformFlow { velocity } => *velocity,
            
            ExternalField::Rotation { angular_velocity, center } => {
                let r = position - center;
                angular_velocity.cross(&r)
            },
            
            ExternalField::OscillatoryFlow { amplitude, frequency, phase } => {
                let omega = 2.0 * std::f64::consts::PI * frequency;
                let factor = (omega * time + phase).sin();
                amplitude.scale(factor)
            },
            
            ExternalField::PoiseuilleFlow { max_velocity, axis, radius } => {
                let axis_unit = axis.normalize();
                
                // Project position onto plane perpendicular to axis
                let projected = position - axis_unit.scale(position.dot(&axis_unit));
                let r_squared = projected.norm_squared();
                
                if r_squared > radius * radius {
                    return Vector3::zeros();
                }
                
                // Parabolic velocity profile
                let factor = max_velocity * (1.0 - r_squared / (radius * radius));
                axis_unit.scale(factor)
            },
            
            ExternalField::Counterflow { velocity } => *velocity,
        }
    }
    
    /// Calculate mutual friction contribution from normal fluid
    pub fn mutual_friction(&self, position: &Vector3<f64>, tangent: &Vector3<f64>, time: f64, temperature: f64) -> Vector3<f64> {
        // Only relevant for counterflow and at temperatures above absolute zero
        if temperature < 0.001 {
            return Vector3::zeros();
        }
        
        // Get normal fluid velocity at this point
        let v_n = match self {
            ExternalField::Counterflow { velocity } => *velocity,
            _ => return Vector3::zeros(), // No mutual friction for other fields
        };
        
        // Temperature-dependent mutual friction coefficients
        // Simple approximation based on temperature
        let alpha = 0.1 * temperature / 2.17;
        let alpha_prime = 0.01 * temperature / 2.17;
        
        // Calculate velocity difference between normal and superfluid components
        // For counterflow, we assume superfluid is initially stationary (v_s = 0)
        let v_ns = v_n; // v_n - v_s where v_s = 0
        
        // Tangent vector as Vector3
        let s_prime = *tangent;
        
        // Calculate mutual friction force
        // F_mf = α s' × (s' × v_ns) + α' s' × v_ns
        let s_cross_v = s_prime.cross(&v_ns);
        let s_cross_s_cross_v = s_prime.cross(&s_cross_v);
        
        s_cross_s_cross_v.scale(alpha) + s_cross_v.scale(alpha_prime)
    }
}