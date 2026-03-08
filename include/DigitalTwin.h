#pragma once
#include "DigitalTwinTypes.h"

#include "core/Core.h"
#include "platform/Window.h"
#include "simulation/SimulationBlueprint.h"
#include <string>

namespace DigitalTwin
{

    class DT_API DigitalTwin
    {
    public:
        DigitalTwin();
        ~DigitalTwin();

        Result Initialize( const DigitalTwinConfig& config );
        void   Shutdown();

        /**
         * @brief Sets the blueprint definition for the simulation without allocating GPU resources.
         */
        void SetBlueprint( const SimulationBlueprint& blueprint );

        const FrameContext& BeginFrame();
        void                EndFrame();
        void                Step(); // TODO: For future use

        void Play();
        void Pause();
        void Stop();

        EngineState GetState() const;

        bool IsWindowClosed();

        // Rendering
        void                             SetGridVisualization( const GridVisualizationSettings& settings );
        const GridVisualizationSettings& GetGridVisualization() const;
        void                             RenderUI( std::function<void()> uiCallback );
        void                             SetViewportSize( uint32_t width, uint32_t height );
        void                             GetViewportSize( uint32_t& outWidth, uint32_t& outHeight ) const;
        void*                            GetSceneTextureID() const;
        void*                            GetImGuiTextureID( TextureHandle handle );
        void*                            GetImGuiContext();

    public:
        FileSystem* GetFileSystem() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace DigitalTwin