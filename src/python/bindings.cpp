#include "app/Simulation.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_simulation_config( py::module& m )
{
    py::class_<DigitalTwin::SimulationConfig>( m, "SimulationConfig" )
        .def( py::init<>() )
        .def_readwrite( "maxSteps", &DigitalTwin::SimulationConfig::maxSteps )
        .def( "__repr__",
              []( const DigitalTwin::SimulationConfig& config ) { return "SimulationConfig(maxSteps=" + std::to_string( config.maxSteps ) + ")"; } );
}

void bind_simulation( py::module& m )
{
    py::class_<DigitalTwin::Simulation>( m, "Simulation" )
        .def( py::init<>() )
        .def( "initialize", &DigitalTwin::Simulation::Initialize )
        .def( "step", &DigitalTwin::Simulation::Step )
        .def( "is_complete", &DigitalTwin::Simulation::IsComplete )
        .def( "get_current_step", &DigitalTwin::Simulation::GetCurrentStep )
        .def( "__repr__", []( const DigitalTwin::Simulation& sim ) {
            return "Simulation(current_step=" + std::to_string( sim.GetCurrentStep() ) + ", is_complete=" + std::to_string( sim.IsComplete() ) + ")";
        } );
}

PYBIND11_MODULE( DigitalTwin, m )
{
    m.doc() = "Digital Twin Python bindings";

    bind_simulation_config( m );
    bind_simulation( m );
}