#!/usr/bin/env python3

try:
    import DigitalTwin
except ImportError as e:
    print(f"Error: Cannot import DigitalTwin module")
    sys.exit(1)

def main():
    print("=== CellSim Minimal Demo ===\n")

    sim = DigitalTwin.Simulation()
    sim.initialize()

    while not sim.is_complete():
        sim.step()

    print(f"=== Demo finished after {sim.get_current_step()} steps ===\n")

if __name__ == "__main__":
    main()