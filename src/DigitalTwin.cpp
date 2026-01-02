#include "DigitalTwin.h"

#include <iostream>

namespace DigitalTwin
{

    DigitalTwin::DigitalTwin()
        : m_name( "DigitalTwin Engine v0.1" )
    {
        std::cout << "[DLL] DigitalTwin Initialized." << std::endl;
    }

    DigitalTwin::~DigitalTwin()
    {
        std::cout << "[DLL] DigitalTwin Destroyed." << std::endl;
    }

    void DigitalTwin::Print()
    {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Hello from DLL! Name: " << m_name << std::endl;
        std::cout << "Linker works properly if you see this message." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }

} // namespace DT