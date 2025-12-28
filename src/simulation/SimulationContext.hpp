#pragma once
#include "compute/ComputeKernel.hpp"
#include "core/Base.hpp"
#include "resources/StreamingManager.hpp"
#include "rhi/Buffer.hpp"
#include "rhi/Device.hpp"
#include "simulation/Types.hpp"
#include <array>
#include <vector>

namespace DigitalTwin
{
    class SystemBindings; // Forward decl

    class SimulationContext
    {
    public:
        SimulationContext( Ref<Device> device );
        ~SimulationContext();

        void Init( uint32_t maxCells );
        void Shutdown();
        void UploadState( StreamingManager* streamer, const std::vector<Cell>& cells );

        // --- Double Buffering Logic ---
        void                SwapBuffers();
        Ref<SystemBindings> CreateSystemBindings( Ref<ComputeKernel> kernel );

        // Accessors for SystemBindings (Internal)
        Ref<Buffer> GetBuffer( uint32_t index ) const;

        // --- Public Accessors ---

        // Re-added for Tests/Renderer compatibility (returns stable buffer)
        Ref<Buffer> GetCellBuffer() const;
        Ref<Buffer> GetRenderBuffer() const { return GetCellBuffer(); }

        Ref<Buffer> GetCounterBuffer() const { return m_atomicCounter; }
        uint32_t    GetMaxCellCount() const { return m_maxCellCount; }
        uint32_t    GetFrameIndex() const { return m_frameIndex; }
        Ref<Device> GetDevice() const { return m_device; }

        void SetFrameIndex( uint32_t idx ) { m_frameIndex = idx % 2; }

    private:
        Ref<Device>                m_device;
        std::array<Ref<Buffer>, 2> m_cellBuffers;
        Ref<Buffer>                m_atomicCounter;
        uint32_t                   m_maxCellCount = 0;
        uint32_t                   m_frameIndex   = 0;
    };
} // namespace DigitalTwin