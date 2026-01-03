#include <DigitalTwin.h>
#include <iostream>

int main()
{
    std::cout << "[EXE] Starting Editor..." << std::endl;

    DigitalTwin::DigitalTwin engine;
    DigitalTwin::DigitalTwinConfig config;
    engine.Initialize( config );
    engine.Print();
    engine.Shutdown();

    std::cout << "[EXE] Editor closing..." << std::endl;

    return 0;
}