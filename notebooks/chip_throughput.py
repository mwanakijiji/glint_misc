from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from astropy import units as u
import yaml
import ipdb

# Basic flow:
# read in config file that sets the name of the star and the method of luminosity calculation
# read in photometry or spectral of star
# convert to luminosity (fit to BB?)
# calculate flux incident on Subaru telescope pupil
# read in throughput of Subaru telescope and SCExAO
# find expected photons/sec on detector
# calculate expected signal/sec in ADU
# read in integration parameters
# find flux implied by ADU counts
# read in the number of empirical ADU counts on the detector
# find total throughput

@dataclass
class StarConfig:
    """Configuration for the target star"""
    name: str
    spectral_type: str
    magnitude: float
    band: str  # e.g., 'H'
    temperature: Optional[float] = None  # If fitting blackbody

@dataclass
class TelescopeConfig:
    """Subaru telescope parameters"""
    diameter: float  # meters
    central_obstruction: float  # meters
    throughput: float  # Overall telescope throughput

@dataclass
class DetectorConfig:
    """Detector parameters"""
    quantum_efficiency: float
    gain: float  # e-/ADU
    integration_time: float  # seconds
    dark_current: float  # e-/s
    read_noise: float  # e-

class ThroughputCalculator:
    """Handles flux and throughput calculations"""
    
    def __init__(self, star_config: StarConfig, 
                 telescope_config: TelescopeConfig,
                 detector_config: DetectorConfig):
        self.star = star_config
        self.telescope = telescope_config
        self.detector = detector_config
        
    def calculate_stellar_flux(self) -> float:
        """Calculate incident stellar flux"""
        # Convert magnitude to flux
        # If temperature provided, fit to blackbody
        # Return photons/s/m²
        pass
    
    def calculate_telescope_flux(self) -> float:
        """Calculate flux at telescope pupil"""
        # Calculate effective collecting area
        area = np.pi * (self.telescope.diameter/2)**2 - \
               np.pi * (self.telescope.central_obstruction/2)**2
        
        # Multiply by stellar flux and telescope throughput
        return self.calculate_stellar_flux() * area * self.telescope.throughput
    
    def calculate_expected_counts(self) -> float:
        """Calculate expected detector counts in ADU"""
        # Convert photon flux to electrons
        electrons = (self.calculate_telescope_flux() * 
                    self.detector.quantum_efficiency * 
                    self.detector.integration_time)
        
        # Add dark current and read noise
        electrons += (self.detector.dark_current * self.detector.integration_time + 
                     self.detector.read_noise)
        
        # Convert to ADU
        return electrons / self.detector.gain
    
    def calculate_throughput(self, measured_counts: float) -> float:
        """Calculate total system throughput"""
        expected_counts = self.calculate_expected_counts()
        return measured_counts / expected_counts

class DataLoader:
    """Handles loading of configuration and measurement data"""
    
    @staticmethod
    def load_config(config_file: str) -> Dict:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_measurements(data_file: str) -> float:
        """Load measured detector counts"""
        # Implementation depends on data format
        pass

def main(config_file: str):
    """Example usage"""
    # Load configurations
    config = DataLoader.load_config(config_file)
    
    # Create configuration objects
    star_config = StarConfig(**config['star'])
    telescope_config = TelescopeConfig(**config['telescope'])
    detector_config = DetectorConfig(**config['detector'])
    ipdb.set_trace()
    
    # Initialize calculator
    calculator = ThroughputCalculator(star_config, telescope_config, detector_config)
    
    # Load measured counts
    measured_counts = DataLoader.load_measurements('measurements.dat')
    
    # Calculate throughput
    throughput = calculator.calculate_throughput(measured_counts)
    
    # Print results
    print(f"Expected counts: {calculator.calculate_expected_counts():.2f} ADU")
    print(f"Measured counts: {measured_counts:.2f} ADU")
    print(f"Total system throughput: {throughput:.2%}")

if __name__ == "__main__":
    main(config_file = 'config_files/config.yaml')
