#include "renderer/RenderContext.hpp"

#include "core/Log.hpp"
#include "rhi/Queue.hpp"
#include "rhi/RHI.hpp"

namespace DigitalTwin
{
    RenderContext::RenderContext( Ref<Device> device, Window* window )
        : m_device( device )
        , m_window( window )
    {
    }

    RenderContext::~RenderContext()
    {
        Shutdown();
    }

    void RenderContext::Shutdown()
    {
        // Ensure GPU is idle before destroying resources
        if( m_device )
            m_device->GetAPI().vkDeviceWaitIdle( m_device->GetHandle() );

        m_viewportColors.clear();
        m_viewportDepths.clear();

        for( auto& frame: m_frames )
        {
            if( frame.renderFinished )
                m_device->GetAPI().vkDestroySemaphore( m_device->GetHandle(), frame.renderFinished, nullptr );
        }
        m_frames.clear();

        // Explicit reset to ensure correct destruction order
        m_swapchain.reset();
    }

    Result RenderContext::Init()
    {
        // 1. Swapchain Setup
        SwapchainDesc swapDesc;
        swapDesc.width        = m_window->GetWidth();
        swapDesc.height       = m_window->GetHeight();
        swapDesc.vsync        = true;
        swapDesc.windowHandle = m_window->GetNativeWindow();

        auto graphicsQueue = m_device->GetGraphicsQueue();

        m_swapchain = CreateRef<Swapchain>( m_device->GetHandle(), m_device->GetPhysicalDevice(), RHI::GetInstance(), graphicsQueue->GetHandle(),
                                            graphicsQueue->GetFamilyIndex(), &m_device->GetAPI(), swapDesc );

        // 2. Frame Sync Setup
        m_frames.resize( FRAMES_IN_FLIGHT );
        for( auto& frame: m_frames )
        {
            frame.cmd = m_device->CreateCommandBuffer( QueueType::GRAPHICS );

            VkSemaphoreCreateInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            m_device->GetAPI().vkCreateSemaphore( m_device->GetHandle(), &semInfo, nullptr, &frame.renderFinished );

            // Initialize timeline value to 0
            frame.timelineValue = 0;
        }

        // 3. Viewport Sampler
        SamplerDesc samplerDesc  = {};
        samplerDesc.magFilter    = VK_FILTER_LINEAR;
        samplerDesc.minFilter    = VK_FILTER_LINEAR;
        samplerDesc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerDesc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerDesc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        m_viewportSampler        = m_device->CreateSampler( samplerDesc );

        CreateViewportResources( 1280, 720 );

        return Result::SUCCESS;
    }

    void RenderContext::CreateViewportResources( uint32_t width, uint32_t height )
    {
        if( width == 0 || height == 0 )
            return;

        m_viewportColors.clear();
        m_viewportDepths.clear();

        m_viewportColors.resize( FRAMES_IN_FLIGHT );
        m_viewportDepths.resize( FRAMES_IN_FLIGHT );

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            TextureDesc colorDesc = {};
            colorDesc.width       = width;
            colorDesc.height      = height;
            colorDesc.format      = VK_FORMAT_R8G8B8A8_UNORM;
            colorDesc.usage       = TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED;
            m_viewportColors[ i ] = m_device->CreateTexture( colorDesc );

            TextureDesc depthDesc = {};
            depthDesc.width       = width;
            depthDesc.height      = height;
            depthDesc.format      = VK_FORMAT_D32_SFLOAT;
            depthDesc.usage       = TextureUsage::DEPTH_STENCIL_TARGET;
            m_viewportDepths[ i ] = m_device->CreateTexture( depthDesc );
        }

        DT_CORE_INFO( "[RenderContext] Recreated Viewport Resources: {}x{}", width, height );
    }

    bool RenderContext::OnResizeViewport( uint32_t width, uint32_t height )
    {
        if( !m_viewportColors.empty() )
        {
            if( m_viewportColors[ 0 ]->GetExtent().width == width && m_viewportColors[ 0 ]->GetExtent().height == height )
                return false;
        }

        // --- CRITICAL SYNC ---
        // Wait for GPU to finish everything before we destroy textures.
        // This relies on your API: Device::WaitIdle() -> vkDeviceWaitIdle
        m_device->GetAPI().vkDeviceWaitIdle( m_device->GetHandle() );

        CreateViewportResources( width, height );
        return true;
    }

    Ref<CommandBuffer> RenderContext::BeginFrame()
    {
        auto& frame = m_frames[ m_frameIndex ];

        // --- CPU WAIT FOR GPU (Timeline Semaphore) ---
        // If this frame slot was used before, wait until the GPU reached the timeline value
        // assigned at the end of that previous usage.
        if( frame.timelineValue > 0 )
        {
            // Using your Device::WaitForQueue API
            Result res = m_device->WaitForQueue( m_device->GetGraphicsQueue(), frame.timelineValue );
            if( res != Result::SUCCESS )
            {
                DT_CORE_CRITICAL( "Wait For Queue Failed in BeginFrame!" );
                return nullptr;
            }
        }

        // --- Acquire Swapchain Image ---
        uint32_t    imageIndex  = 0;
        VkSemaphore acquiredSem = m_swapchain->AcquireNextImage( imageIndex );

        if( acquiredSem == VK_NULL_HANDLE )
        {
            RecreateSwapchain();
            return nullptr;
        }

        m_imageIndex                = imageIndex;
        frame.currentImageAvailable = acquiredSem;

        // It is now safe to reset the command buffer because we passed WaitForQueue
        frame.cmd->Begin();

        return frame.cmd;
    }

    void RenderContext::EndFrame( const std::vector<VkSemaphore>& waitSemaphores, const std::vector<uint64_t>& waitValues )
    {
        auto& frame = m_frames[ m_frameIndex ];
        frame.cmd->End();

        // Build Submit Info using your helper structs
        SubmitInfo submitInfo = {};
        submitInfo.commandBuffers.push_back( frame.cmd->GetHandle() );

        // 1. Wait for Swapchain Image Available
        if( frame.currentImageAvailable )
        {
            QueueWaitInfo waitInfo = {};
            waitInfo.semaphore     = frame.currentImageAvailable;
            waitInfo.value         = 0; // Binary semaphore
            waitInfo.stageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            submitInfo.waitSemaphores.push_back( waitInfo );
        }

        // 2. Wait for External (Compute/Transfer)
        for( size_t i = 0; i < waitSemaphores.size(); ++i )
        {
            QueueWaitInfo waitInfo = {};
            waitInfo.semaphore     = waitSemaphores[ i ];
            waitInfo.value         = ( i < waitValues.size() ) ? waitValues[ i ] : 0;
            // Wait at start of pipe to ensure resources are ready
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            submitInfo.waitSemaphores.push_back( waitInfo );
        }

        // 3. Signal 'RenderFinished' for Presentation
        {
            QueueSignalInfo sigInfo = {};
            sigInfo.semaphore       = frame.renderFinished;
            sigInfo.value           = 0; // Binary
            sigInfo.stageMask       = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            submitInfo.signalSemaphores.push_back( sigInfo );
        }

        // --- Submit & Track Timeline ---
        // Your Queue::Submit returns the new signal value into the pointer.
        uint64_t newTimelineValue = 0;
        Result   submitRes        = m_device->GetGraphicsQueue()->Submit( submitInfo, &newTimelineValue );

        if( submitRes == Result::SUCCESS )
        {
            // Store this value so we can wait for it next time we use 'm_frameIndex'
            frame.timelineValue = newTimelineValue;
        }
        else
        {
            DT_CORE_ERROR( "Queue Submit Failed!" );
        }

        // --- Present ---
        m_swapchain->Present( frame.renderFinished );

        // Advance Frame Index
        m_frameIndex = ( m_frameIndex + 1 ) % FRAMES_IN_FLIGHT;
    }

    void RenderContext::OnResizeSwapchain( uint32_t width, uint32_t height )
    {
        if( width == 0 || height == 0 )
            return;
        RecreateSwapchain();
    }

    void RenderContext::RecreateSwapchain()
    {
        m_device->GetAPI().vkDeviceWaitIdle( m_device->GetHandle() );
        m_swapchain->Resize( m_window->GetWidth(), m_window->GetHeight() );
    }
} // namespace DigitalTwin