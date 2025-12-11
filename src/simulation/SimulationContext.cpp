#include "simulation/SimulationContext.hpp"

#include "core/Log.hpp"

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

        // 1. Allocate Main Cell Buffer (SSBO)
        VkDeviceSize bufferSize = m_maxCellCount * sizeof( Cell );

        BufferDesc cellDesc{};
        cellDesc.size = bufferSize;
        cellDesc.type = BufferType::STORAGE;
        m_cellBuffer  = m_device->CreateBuffer( cellDesc );
        if( !m_cellBuffer )
        {
            DT_CORE_CRITICAL( "[Simulation] Failed to allocate Cell Buffer!" );
            return;
        }

        // 2. Allocate Atomic Counter (4 bytes)
        // This holds the single uint32_t count of living agents.
        BufferDesc counterDesc{};
        counterDesc.size = sizeof( uint32_t );
        counterDesc.type = BufferType::ATOMIC_COUNTER;
        m_atomicCounter  = m_device->CreateBuffer( counterDesc );
        if( !m_atomicCounter )
        {
            DT_CORE_CRITICAL( "[Simulation] Failed to allocate Atomic Counter!" );
            return;
        }

        DT_CORE_INFO( "[Simulation] Context Initialized. Capacity: {0} cells ({1:.2f} MB). Counter created.", m_maxCellCount,
                      bufferSize / ( 1024.f * 1024.f ) );
    }

    void SimulationContext::UploadState( StreamingManager* streamer, const std::vector<Cell>& cells )
    {
        if( !m_cellBuffer || !m_atomicCounter )
            return;

        if( cells.size() > m_maxCellCount )
        {
            DT_CORE_ERROR( "[Simulation] Upload count ({0}) exceeds capacity ({1})!", cells.size(), m_maxCellCount );
            return;
        }

        // 1. Upload Cell Data
        VkDeviceSize dataSize = cells.size() * sizeof( Cell );
        if( dataSize > 0 )
        {
            streamer->UploadToBuffer( m_cellBuffer, cells.data(), dataSize, 0 );
        }

        // 2. Update Atomic Counter to match initial count
        uint32_t initialCount = static_cast<uint32_t>( cells.size() );

        // We upload the count (4 bytes) to the atomic counter buffer
        streamer->UploadToBuffer( m_atomicCounter, &initialCount, sizeof( uint32_t ), 0 );

        DT_CORE_INFO( "[Simulation] State uploaded. Active cells: {0}", initialCount );
    }

    void SimulationContext::Shutdown()
    {
        m_cellBuffer.reset();
        m_atomicCounter.reset();
        m_maxCellCount = 0;
    }
} // namespace DigitalTwin