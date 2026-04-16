#pragma once
#include "EditorPanel.h"
#include <functional>

namespace DigitalTwin
{
    class DigitalTwin;
}

namespace Gaudi
{
    /**
     * @brief Extensible render-settings panel.
     *
     * Current sections:
     *   - Anti-Aliasing  (Off / 4x MSAA radio — 4x greyed out when unsupported)
     *   - Present        (V-Sync toggle)
     *
     * Adding a new section is one line: DrawSection("Name", [&](){ ... });
     */
    class RenderSettingsPanel : public EditorPanel
    {
    public:
        explicit RenderSettingsPanel( DigitalTwin::DigitalTwin& engine );
        ~RenderSettingsPanel() override = default;

        void OnUIRender() override;

        bool IsVisible() const { return m_visible; }
        void SetVisible( bool v ) { m_visible = v; }

    private:
        static void DrawSection( const char* label, std::function<void()> body );

        DigitalTwin::DigitalTwin& m_engine;
        bool                      m_visible = false;
    };
} // namespace Gaudi
