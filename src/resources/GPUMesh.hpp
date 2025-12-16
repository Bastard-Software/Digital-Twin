#pragma once
#include "core/Base.hpp"
#include "rhi/Buffer.hpp"
#include "rhi/Device.hpp"

namespace DigitalTwin
{
    /**
     * @brief A mesh residing on the GPU in a SINGLE merged buffer.
     * Layout in memory: [Vertices... | Indices...]
     */
    class GPUMesh
    {
    public:
        /**
         * @brief Constructs a GPUMesh from a single merged buffer.
         * @param buffer The unified buffer containing both vertices and indices.
         * @param indexOffsetBytes Byte offset where the Index section starts.
         * @param indexCount Number of indices.
         */
        GPUMesh( Ref<Buffer> buffer, VkDeviceSize indexOffsetBytes, uint32_t indexCount )
            : m_buffer( buffer )
            , m_indexOffset( indexOffsetBytes )
            , m_indexCount( indexCount )
        {
        }

        // Returns the unified buffer handle
        Ref<Buffer> GetBuffer() const { return m_buffer; }

        // Returns the byte offset where indices start (needed for vkCmdBindIndexBuffer)
        VkDeviceSize GetIndexOffset() const { return m_indexOffset; }

        // Returns the number of indices to draw
        uint32_t GetIndexCount() const { return m_indexCount; }

    private:
        Ref<Buffer>  m_buffer;
        VkDeviceSize m_indexOffset = 0;
        uint32_t     m_indexCount  = 0;
    };
} // namespace DigitalTwin