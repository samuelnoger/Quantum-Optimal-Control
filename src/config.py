# src/config.py
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareConfig:
    anharmonicity_mhz: float
    levels: int
    coupling_mhz: Optional[float] = None
    idle_detuning_mhz: Optional[float] = None
    n_shots: Optional[int] = None

@dataclass
class DecoherenceConfig:
    t1_ns: float
    t2_ns: float

@dataclass
class PulseConfig:
    duration_ns: float
    dt_ns: float
    max_amp_mhz: Optional[float] = None
    max_flux_amp_ghz: Optional[float] = None
    cutoff_freq_ghz: Optional[float] = None

@dataclass
class OptimizerConfig:
    learning_rate: float
    steps: int
    leakage_weight: float

@dataclass
class ExperimentConfig:
    hardware: HardwareConfig
    pulse: PulseConfig
    optimizer: OptimizerConfig
    decoherence: Optional[DecoherenceConfig] = None

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        # Resolve the absolute path from the project root
        project_root = Path(__file__).parent.parent
        absolute_path = project_root / filepath
        
        with open(absolute_path, 'r') as f:
            raw = yaml.safe_load(f)
            
        # Safely handle the decoherence block if it exists
        decoherence_data = raw.get('decoherence')
        decoherence_config = DecoherenceConfig(**decoherence_data) if decoherence_data else None
            
        return cls(
            hardware=HardwareConfig(**raw['hardware']),
            pulse=PulseConfig(**raw['pulse']),
            optimizer=OptimizerConfig(**raw['optimizer']),
            decoherence=decoherence_config
        )