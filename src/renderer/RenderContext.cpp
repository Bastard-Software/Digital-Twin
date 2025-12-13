#include "renderer/RenderContext.hpp"

#include "core/Log.hpp"

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
        if( m_device )
            m_device->GetAPI().vkDeviceWaitIdle( m_device->GetHandle() );

        for( auto& frame: m_frames )
        {
            // Only destroy what we created. imageAvailable belongs to Swapchain.
            if( frame.renderFinished )
                m_device->GetAPI().vkDestroySemaphore( m_device->GetHandle(), frame.renderFinished, nullptr );

            frame.renderFinished        = VK_NULL_HANDLE;
            frame.currentImageAvailable = VK_NULL_HANDLE;
            frame.timelineValue         = 0;
            frame.cmd.reset(); // Release CommandBuffer wrapper
        }
    }

    Result RenderContext::Init()
    {
        SwapchainDesc swapDesc;
        swapDesc.width        = m_window->GetWidth();
        swapDesc.height       = m_window->GetHeight();
        swapDesc.vsync        = true;
        swapDesc.windowHandle = m_window->GetNativeWindow();

        m_swapchain = m_device->CreateSwapchain( swapDesc );

        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            // Create RenderFinished binary semaphore (needed for Present)
            VkSemaphoreCreateInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            if( m_device->GetAPI().vkCreateSemaphore( m_device->GetHandle(), &semInfo, nullptr, &m_frames[ i ].renderFinished ) != VK_SUCCESS )
            {
                DT_CORE_CRITICAL( "Failed to create render semaphore!" );
                return Result::FAIL;
            }

            m_frames[ i ].cmd           = m_device->CreateCommandBuffer( QueueType::GRAPHICS );
            m_frames[ i ].timelineValue = 0;
        }
        return Result::SUCCESS;
    }

    CommandBuffer* RenderContext::BeginFrame()
    {
        auto& frame         = m_frames[ m_frameIndex ];
        auto  graphicsQueue = m_device->GetGraphicsQueue();

        // 1. CPU Wait: Wait for the previous usage of this frame slot to finish on GPU
        // We use the timeline value returned by the last Submit on this slot.
        if( frame.timelineValue > 0 )
        {
            if( m_device->WaitForQueue( graphicsQueue, frame.timelineValue ) != Result::SUCCESS )
            {
                DT_CORE_ERROR( "RenderContext: Timeout waiting for frame timeline!" );
                return nullptr;
            }
        }

        // 2. Acquire Image from Swapchain
        // This returns the semaphore that will be signalled when the image is ready.
        VkSemaphore acquireSem = m_swapchain->AcquireNextImage( m_imageIndex );

        if( acquireSem == VK_NULL_HANDLE )
        {
            // Swapchain indicates resize needed or error
            // Assuming Swapchain implementation might handle internal resize logic or return null
            return nullptr;
        }

        frame.currentImageAvailable = acquireSem;

        // 3. Begin Command Buffer
        // (Wrapper implicitly handles reset if using proper pool flags)
        frame.cmd->Begin();

        // 4. Transition Swapchain Image: Undefined -> Color Attachment
        // We need the image handle. Swapchain exposes GetImage(index).
        frame.cmd->TransitionImageLayout( m_swapchain->GetImage( m_imageIndex ), VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL );

        return frame.cmd.get();
    }

    void RenderContext::EndFrame( const std::vector<VkSemaphore>& waitSemaphores, const std::vector<uint64_t>& waitValues )
    {
        auto& frame = m_frames[ m_frameIndex ];

        // 1. Transition: Color Attachment -> Present
        frame.cmd->TransitionImageLayout( m_swapchain->GetImage( m_imageIndex ), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR );
        frame.cmd->End();

        // 2. Build Submit Info
        SubmitInfo submitInfo;
        submitInfo.commandBuffers.push_back( frame.cmd->GetHandle() );

        // Wait: Swapchain Image Available (Binary)
        // Stage: Color Output (we can run Vertex shader before image is available)
        submitInfo.waitSemaphores.push_back( { frame.currentImageAvailable, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT } );

        // Wait: External (Compute) Timeline Semaphores
        for( size_t i = 0; i < waitSemaphores.size(); ++i )
        {
            submitInfo.waitSemaphores.push_back( { waitSemaphores[ i ], waitValues[ i ], VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT } );
        }

        // Signal: Render Finished (Binary) -> Needed for Present
        submitInfo.signalSemaphores.push_back( { frame.renderFinished, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT } );

        // 3. Submit
        // The queue will increment its internal timeline and return the value in frame.timelineValue
        m_device->GetGraphicsQueue()->Submit( submitInfo, &frame.timelineValue );

        // 4. Present
        m_swapchain->Present( frame.renderFinished );

        // Move to next frame slot
        m_frameIndex = ( m_frameIndex + 1 ) % FRAMES_IN_FLIGHT;
    }

    void RenderContext::OnResize( uint32_t width, uint32_t height )
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