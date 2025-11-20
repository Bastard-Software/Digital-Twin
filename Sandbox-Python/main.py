#!/usr/bin/env python3

import sys
import os

# Dodaj ścieżkę do zbudowanego modułu Pythonowego
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
python_packages_dir = os.path.join(build_dir, 'python_packages')

sys.path.insert(0, python_packages_dir)

try:
    import digital_twin
except ImportError as e:
    print(f"Error: Cannot import digital_twin module")
    print(f"Make sure you have built the project and the module is in: {python_packages_dir}")
    sys.exit(1)

def main():
    print("=== CellSim Minimal Demo ===\n")

    sim = digital_twin.Simulation()
    sim.initialize()

    while not sim.is_complete():
        sim.step()

    print(f"=== Demo finished after {sim.get_current_step()} steps ===\n")

if __name__ == "__main__":
    main()