#include "simulation/SimulationContext.hpp"

#include "core/Log.hpp"
#include "simulation/SystemBindings.hpp" // FULL DEFINITION NEEDED HERE

namespace DigitalTwin
{
    SimulationContext::SimulationContext( Ref<Device> device )
        : m_device( device )
    {
    }
    SimulationContext::~SimulationContext()
    {
        Shutdown();
    }

    void SimulationContext::Init( uint32_t maxCells )
    {
        m_maxCellCount = maxCells;
        if( m_maxCellCount == 0 )
            return;

        VkDeviceSize bufferSize = m_maxCellCount * sizeof( Cell );
        BufferDesc   cellDesc{ bufferSize, BufferType::STORAGE };

        m_cellBuffers[ 0 ] = m_device->CreateBuffer( cellDesc );
        m_cellBuffers[ 1 ] = m_device->CreateBuffer( cellDesc );

        if( !m_cellBuffers[ 0 ] || !m_cellBuffers[ 1 ] )
        {
            DT_CORE_CRITICAL( "[Simulation] Failed to allocate buffers!" );
            return;
        }

        BufferDesc cntDesc{ sizeof( uint32_t ), BufferType::ATOMIC_COUNTER };
        m_atomicCounter = m_device->CreateBuffer( cntDesc );

        DT_CORE_INFO( "[Simulation] Context Init. Capacity: {}", m_maxCellCount );
    }

    void SimulationContext::UploadState( StreamingManager* streamer, const std::vector<Cell>& cells )
    {
        if( !m_cellBuffers[ 0 ] )
            return;
        VkDeviceSize dataSize = cells.size() * sizeof( Cell );
        if( dataSize > 0 )
        {
            streamer->UploadToBuffer( m_cellBuffers[ 0 ], cells.data(), dataSize, 0 );
            streamer->UploadToBuffer( m_cellBuffers[ 1 ], cells.data(), dataSize, 0 );
        }
        uint32_t count = ( uint32_t )cells.size();
        streamer->UploadToBuffer( m_atomicCounter, &count, sizeof( uint32_t ), 0 );
    }

    void SimulationContext::SwapBuffers()
    {
        m_frameIndex = 1 - m_frameIndex;
    }

    Ref<SystemBindings> SimulationContext::CreateSystemBindings( Ref<ComputeKernel> kernel )
    {
        return CreateRef<SystemBindings>( this, kernel );
    }

    Ref<Buffer> SimulationContext::GetBuffer( uint32_t index ) const
    {
        return m_cellBuffers[ index % 2 ];
    }

    Ref<Buffer> SimulationContext::GetCellBuffer() const
    {
        return m_cellBuffers[ m_frameIndex ];
    }

    void SimulationContext::Shutdown()
    {
        m_cellBuffers[ 0 ].reset();
        m_cellBuffers[ 1 ].reset();
        m_atomicCounter.reset();
    }
} // namespace DigitalTwin