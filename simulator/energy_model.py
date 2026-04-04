"""
GreenCloudRL - Energy Consumption Model
Linear power model with DVFS support and carbon intensity tracking.
"""

import numpy as np
from typing import List, Optional


class EnergyModel:
    """
    Models power consumption and energy tracking for the data center.
    
    Power model: P(u) = P_idle + (P_max - P_idle) * u * dvfs_factor
    Energy: E = integral(P(u(t)) * dt) for all active servers
    """

    def __init__(
        self,
        power_idle: float = 120.0,
        power_max: float = 200.0,
        pue: float = 1.4,               # Power Usage Effectiveness
        carbon_intensity: float = 0.5,   # kgCO2/kWh (average grid)
    ):
        self.power_idle = power_idle
        self.power_max = power_max
        self.pue = pue
        self.carbon_intensity = carbon_intensity

        # Tracking
        self.total_energy_joules = 0.0
        self.total_carbon_kg = 0.0
        self.energy_history = []
        self.power_history = []

    def compute_power(self, utilization: float, dvfs: float = 1.0) -> float:
        """
        Compute instantaneous power for a single server.
        
        Args:
            utilization: CPU utilization in [0, 1]
            dvfs: DVFS frequency scaling factor
            
        Returns:
            Power in Watts
        """
        u = np.clip(utilization, 0.0, 1.0)
        return self.power_idle + (self.power_max - self.power_idle) * u * dvfs

    def compute_datacenter_power(self, servers) -> float:
        """
        Compute total data center power including PUE overhead.
        
        Args:
            servers: List of Server objects
            
        Returns:
            Total power in Watts (including cooling, networking overhead)
        """
        it_power = sum(s.current_power() for s in servers if s.is_active)
        return it_power * self.pue

    def update(self, servers, dt: float):
        """
        Update energy and carbon tracking for a time step.
        
        Args:
            servers: List of Server objects
            dt: Time step in seconds
        """
        total_power = self.compute_datacenter_power(servers)
        energy_j = total_power * dt
        energy_kwh = energy_j / 3_600_000.0

        self.total_energy_joules += energy_j
        self.total_carbon_kg += energy_kwh * self.carbon_intensity

        self.power_history.append(total_power)
        self.energy_history.append(self.total_energy_joules)

    @property
    def total_energy_kwh(self) -> float:
        return self.total_energy_joules / 3_600_000.0

    def get_energy_cost(self, price_per_kwh: float = 0.12) -> float:
        """Calculate total energy cost in dollars."""
        return self.total_energy_kwh * price_per_kwh

    def reset(self):
        """Reset all tracking."""
        self.total_energy_joules = 0.0
        self.total_carbon_kg = 0.0
        self.energy_history.clear()
        self.power_history.clear()

    def get_stats(self) -> dict:
        """Return energy statistics."""
        return {
            "total_energy_joules": self.total_energy_joules,
            "total_energy_kwh": self.total_energy_kwh,
            "total_carbon_kg": self.total_carbon_kg,
            "avg_power_watts": np.mean(self.power_history) if self.power_history else 0.0,
            "peak_power_watts": max(self.power_history) if self.power_history else 0.0,
        }
