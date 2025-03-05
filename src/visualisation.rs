use crate::simulation::VortexSimulation;
use std::fs::File;
use std::io::{self, Write};

// Save results in VTK format for visualization
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
    let mut line_size = 0;
    for count in &line_counts {
        line_size += count + 1; // Add 1 for the count value at the start of each line
    }
    
    writeln!(file, "LINES {} {}", line_counts.len(), line_size)?;
    
    let mut point_offset = 0;
    for count in &line_counts {
        write!(file, "{}", count)?;
        
        for i in 0..*count {
            write!(file, " {}", point_offset + i)?;
        }
        writeln!(file)?;
        
        point_offset += count;
    }
    
    // Add point data for visualization
    writeln!(file, "POINT_DATA {}", total_points)?;
    
    // Tangent vectors
    writeln!(file, "VECTORS tangent float")?;
    
    for line in &simulation.vortex_lines {
        for point in &line.points {
            writeln!(file, "{} {} {}", 
                point.tangent[0], 
                point.tangent[1], 
                point.tangent[2])?;
        }
    }
    
    Ok(())
}

// Save statistics plot
pub fn save_statistics_plot(simulation: &VortexSimulation, filename: &str) -> io::Result<()> {
    let stats = &simulation.stats;
    
    // Simple CSV file for plotting with external tools
    let mut file = File::create(filename)?;
    
    // Header
    writeln!(file, "time,total_length,kinetic_energy")?;
    
    // Data
    for i in 0..stats.time_points.len() {
        writeln!(file, "{},{},{}", 
            stats.time_points[i],
            stats.total_length.get(i).unwrap_or(&0.0),
            stats.kinetic_energy.get(i).unwrap_or(&0.0))?;
    }
    
    Ok(())
}