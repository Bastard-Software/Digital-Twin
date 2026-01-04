#include "core/Log.h"
#include <gtest/gtest.h>

int main( int argc, char** argv )
{
    ::testing::InitGoogleTest( &argc, argv );

    DigitalTwin::Log::Init();

    return RUN_ALL_TESTS();
}