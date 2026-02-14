#pragma once

#include "core/Core.h"
#include "core/Handle.h"

namespace DigitalTwin
{
    class Log;
    class FileSystem;

    DEFINE_HANDLE( TextureHandle );
    DEFINE_HANDLE( BufferHandle );
    DEFINE_HANDLE( ShaderHandle );
    DEFINE_HANDLE( SamplerHandle );
    DEFINE_HANDLE( MeshHandle );
    DEFINE_HANDLE( ComputePipelineHandle );
    DEFINE_HANDLE( GraphicsPipelineHandle );
    DEFINE_HANDLE( ThreadContextHandle );
    DEFINE_HANDLE( CommandBufferHandle );
    DEFINE_HANDLE( BindingGroupHandle );

    enum class GPUType
    {
        DEFAULT,
        DISCRETE,
        INTEGRATED,
    };

    struct DigitalTwinConfig
    {
        GPUType     gpuType       = GPUType::DEFAULT;
        bool_t      headless      = true;
        const char* rootDirectory = nullptr;
        bool_t      debugMode     = false;
        uint32_t    windowWidth   = 1280;
        uint32_t    windowHeight  = 720;
        const char* windowTitle   = "Digital Twin Application";
    };

    struct FrameContext
    {
        CommandBufferHandle graphicsCmd;
        CommandBufferHandle computeCmd;

        TextureHandle sceneTexture;

        BufferHandle agentBufferRead;
        BufferHandle agentBufferWrite;

        uint32_t frameIndex;
    };

} // namespace DigitalTwin