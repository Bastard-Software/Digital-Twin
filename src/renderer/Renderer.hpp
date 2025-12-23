#pragma once
#include "core/Base.hpp"
#include "renderer/AgentRenderPass.hpp"
#include "renderer/Camera.hpp"
#include "renderer/RenderContext.hpp"
#include "renderer/Scene.hpp"
#include "rhi/Device.hpp"

namespace DigitalTwin
{
    class Engine;
    class ResourceManager;

    /**
     * @brief High-level visualization facade.
     */
    class Renderer
    {
    public:
        Renderer( Engine& engine );
        ~Renderer();

        /**
         * @brief Renders the scene.
         * Safe to call in headless mode (no-op).
         */
        void Render( const Scene& scene, const std::vector<VkSemaphore>& waitSemaphores = {}, const std::vector<uint64_t>& waitValues = {} );

        void OnUpdate( float dt );
        void OnResize( uint32_t width, uint32_t height );

        Camera& GetCamera() { return *m_camera; }
        bool    IsActive() const { return m_active; }

    private:
        bool                   m_active = false;
        Scope<RenderContext>   m_ctx;
        Scope<AgentRenderPass> m_simPass;
        Scope<Camera>          m_camera;
        Ref<ResourceManager>   m_resManager;
    };
} // namespace DigitalTwin