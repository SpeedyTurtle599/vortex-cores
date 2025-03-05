use crate::simulation::VortexSimulation;
use std::fs::File;
use std::io::{self, Write};

pub fn save_vtk(simulation: &VortexSimulation, filename: &str) -> io::Result<()> {
    let mut file = File::create(filename)?;
    
    // VTK file header
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "Superfluid Helium Vortex Simulation")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET POLYDATA")?;
    
    // Count total points and lines
    let mut total_points = 0;
    let mut line_counts = Vec::new();
    
    for line in &simulation.vortex_lines {
        total_points += line.points.len();
        line_counts.push(line.points.len());
    }
    
    // Write point coordinates
    writeln!(file, "POINTS {} float", total_points)?;
    for line in &simulation.vortex_lines {
        for point in &line.points {
            writeln!(file, "{} {} {}", 
                point.position[0], 
                point.position[1], 
                point.position[2])?;
        }
    }
    
    // Write line connectivity
    let mut line_size = total_points + simulation.vortex_lines.len();
    writeln!(file, "LINES {} {}", simulation.vortex_lines.len(), line_size)?;
    
    let mut point_offset = 0;
    for count in &line_counts {
        write!(file, "{}", count)?;
        
        for i in 0..*count {
            write!(file, " {}", point_offset + i)?;
        }
        writeln!(file)?;
        
        point_offset += count;
    }
    
    Ok(())
}