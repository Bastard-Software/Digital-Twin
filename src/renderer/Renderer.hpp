#pragma once
#include "core/Base.hpp"
#include "renderer/AgentRenderPass.hpp"
#include "renderer/Camera.hpp"
#include "renderer/ImGuiLayer.hpp"
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

        void RenderSimulation( const Scene& scene );
        void RenderUI( const std::vector<VkSemaphore>& waitSemaphores = {}, const std::vector<uint64_t>& waitValues = {} );

        void OnUpdate( float dt );
        void ResizeViewport( uint32_t width, uint32_t height );

        // Getters
        Camera&        GetCamera() { return *m_camera; }
        RenderContext* GetContext() { return m_ctx.get(); }
        ImGuiLayer*    GetGui() const { return m_gui.get(); }

        ImTextureID GetViewportTextureID();

        bool IsActive() const { return m_active; }

    private:
        void UpdateImGuiTextures();

    private:
        bool                   m_active = false;
        Scope<RenderContext>   m_ctx;
        Scope<AgentRenderPass> m_simPass;
        Scope<Camera>          m_camera;
        Ref<ResourceManager>   m_resManager;

        Scope<ImGuiLayer>        m_gui;
        std::vector<ImTextureID> m_viewportTextureIDs;
    };
} // namespace DigitalTwin