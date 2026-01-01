"""Test data simulator"""
import sys
sys.path.append('d:/soham/crisis_forecasting')

from data_sources.data_simulation import DataSimulator

print("Testing Data Simulator...")
sim = DataSimulator()
data = sim.generate_all_data()

print("\n✓ Data simulation successful!\n")
print(f"Climate data: {len(data['climate'])} records")
print(f"Health data: {len(data['health'])} records")
print(f"Food data: {len(data['food'])} records")
print(f"Economic data: {len(data['economic'])} records")

print("\nSample climate data:")
print(data['climate'].head())
