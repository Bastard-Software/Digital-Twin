#include <DigitalTwin.h>
#include <iostream>

int main()
{
    std::cout << "[EXE] Starting Editor..." << std::endl;

    DigitalTwin::DigitalTwin engine;

    engine.Print();

    std::cout << "[EXE] Editor closing..." << std::endl;

    return 0;
}