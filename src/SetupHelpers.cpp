#include "SetupHelpers.h"
#include "core/Log.h"

namespace DigitalTwin::Helpers
{
    std::filesystem::path FindProjectRoot()
    {
        std::filesystem::path p = std::filesystem::current_path();

        for( int i = 0; i < 10; ++i )
        {
            bool hasCMake = std::filesystem::exists( p / "CMakeLists.txt" );
            bool isBuild  = std::filesystem::exists( p / "CMakeFiles" ) || std::filesystem::exists( p / "CMakeCache.txt" );

            if( hasCMake && !isBuild )
                return p;

            std::string pathStr  = p.generic_string();
            size_t      buildPos = pathStr.find( "/build/" );
            if( buildPos != std::string::npos )
            {
                std::string newPathStr = pathStr;
                newPathStr.replace( buildPos, 7, "/" );
                std::filesystem::path trySource = newPathStr;
                if( std::filesystem::exists( trySource ) )
                    return trySource;
            }

            if( p.has_parent_path() )
                p = p.parent_path();
            else
                break;
        }

        return std::filesystem::current_path();
    }

    std::filesystem::path FindEngineRoot( const std::filesystem::path& projectRoot )
    {
        std::filesystem::path p = projectRoot;

        for( int i = 0; i < 10; ++i )
        {
            if( std::filesystem::exists( p / "src" ) && std::filesystem::exists( p / "include" ) && std::filesystem::exists( p / "assets" ) )
                return p;

            if( p.has_parent_path() )
                p = p.parent_path();
            else
                break;
        }
        return {};
    }

    uint32_t SelectGPU( GPUType preference, const std::vector<AdapterInfo>& adapters )
    {
        if( adapters.empty() )
            return 0;

        if( preference == GPUType::DISCRETE )
        {
            for( uint32_t i = 0; i < adapters.size(); ++i )
            {
                if( adapters[ i ].IsDiscrete() )
                {
                    DT_INFO( "GPU Selection: Forced DISCRETE. Found: {0}", adapters[ i ].name );
                    return i;
                }
            }
            DT_WARN( "GPU Selection: Preferred DISCRETE GPU not found. Falling back." );
        }
        else if( preference == GPUType::INTEGRATED )
        {
            for( uint32_t i = 0; i < adapters.size(); ++i )
            {
                if( adapters[ i ].type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU )
                {
                    DT_INFO( "GPU Selection: Forced INTEGRATED. Found: {0}", adapters[ i ].name );
                    return i;
                }
            }
            DT_WARN( "GPU Selection: Preferred INTEGRATED GPU not found. Falling back." );
        }

        int      bestDiscreteIndex   = -1;
        uint64_t maxDiscreteVRAM     = 0;
        int      bestIntegratedIndex = -1;
        uint64_t maxIntegratedVRAM   = 0;

        for( uint32_t i = 0; i < adapters.size(); ++i )
        {
            const auto& info = adapters[ i ];
            if( info.IsDiscrete() )
            {
                if( info.deviceMemorySize >= maxDiscreteVRAM )
                {
                    maxDiscreteVRAM   = info.deviceMemorySize;
                    bestDiscreteIndex = ( int )i;
                }
            }
            else
            {
                if( info.deviceMemorySize >= maxIntegratedVRAM )
                {
                    maxIntegratedVRAM   = info.deviceMemorySize;
                    bestIntegratedIndex = ( int )i;
                }
            }
        }

        if( bestDiscreteIndex != -1 )
        {
            DT_INFO( "GPU Selection: DEFAULT -> Selected Best Discrete: {0}", adapters[ bestDiscreteIndex ].name );
            return ( uint32_t )bestDiscreteIndex;
        }

        if( bestIntegratedIndex != -1 )
        {
            DT_INFO( "GPU Selection: DEFAULT -> Selected Best Integrated: {0}", adapters[ bestIntegratedIndex ].name );
            return ( uint32_t )bestIntegratedIndex;
        }

        DT_WARN( "GPU Selection: Logic failed to find optimal GPU. Returning index 0." );
        return 0;
    }
} // namespace DigitalTwin::Helpers