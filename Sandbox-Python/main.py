#!/usr/bin/env python3

import DigitalTwin as dt
import faulthandler

def main():
    print("=== CellSim Minimal Demo ===\n")

    config = dt.EngineConfig()
    config.headless = False
    dt.Engine.initialize(config)

    sim = dt.Simulation()
    sim.initialize()

    while not sim.is_complete():
        sim.step()

    print(f"=== Demo finished after {sim.get_current_step()} steps ===\n")

if __name__ == "__main__":
    faulthandler.enable()
    main()