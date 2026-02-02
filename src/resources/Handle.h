#pragma once
#include <cstdint>
#include <functional>

#include "core/Handle.h"

namespace DigitalTwin
{

    // Define all handles types here
    DEFINE_HANDLE( TextureHandle );
    DEFINE_HANDLE( BufferHandle );
    DEFINE_HANDLE( ShaderHandle );
    DEFINE_HANDLE( SamplerHandle );
    DEFINE_HANDLE( MeshHandle );
    DEFINE_HANDLE( ComputePipelineHandle );
    DEFINE_HANDLE( GraphicsPipelineHandle );
    DEFINE_HANDLE( ThreadContextHandle );
    DEFINE_HANDLE( CommandBufferHandle );

} // namespace DigitalTwin
