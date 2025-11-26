#include "app/Simulation.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bind_simulation_config( nb::module_& m )
{
    nb::class_<DigitalTwin::SimulationConfig>( m, "SimulationConfig" )
        .def( nb::init<>() )
        .def_rw( "maxSteps", &DigitalTwin::SimulationConfig::maxSteps )
        .def( "__repr__",
              []( const DigitalTwin::SimulationConfig& config ) { return "SimulationConfig(maxSteps=" + std::to_string( config.maxSteps ) + ")"; } );
}

void bind_simulation( nb::module_& m )
{
    nb::class_<DigitalTwin::Simulation>( m, "Simulation" )
        .def( nb::init<>() )
        .def( "initialize", &DigitalTwin::Simulation::Init )
        .def( "step", &DigitalTwin::Simulation::Step )
        .def( "is_complete", &DigitalTwin::Simulation::IsComplete )
        .def( "get_current_step", &DigitalTwin::Simulation::GetCurrentStep )
        .def( "__repr__", []( const DigitalTwin::Simulation& sim ) {
            return "Simulation(current_step=" + std::to_string( sim.GetCurrentStep() ) + ", is_complete=" + std::to_string( sim.IsComplete() ) + ")";
        } );
}

NB_MODULE( DigitalTwin, m )
{
    m.doc() = "Digital Twin Python bindings";

    bind_simulation_config( m );
    bind_simulation( m );
}