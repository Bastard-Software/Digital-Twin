#pragma once
#include "EditorPanel.h"

namespace DigitalTwin
{
    class DigitalTwin;
}

namespace Gaudi
{
    class ViewportPanel : public EditorPanel
    {
    public:
        ViewportPanel( DigitalTwin::DigitalTwin& engine );
        ~ViewportPanel() override = default;

        void OnUIRender() override;

    private:
        DigitalTwin::DigitalTwin& m_engine;
    };
} // namespace Gaudi