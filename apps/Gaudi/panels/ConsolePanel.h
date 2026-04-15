#pragma once
#include "EditorPanel.h"

#include <chrono>
#include <memory>
#include <vector>
#include <spdlog/sinks/ringbuffer_sink.h>

namespace Gaudi
{
    class ConsolePanel : public EditorPanel
    {
    public:
        explicit ConsolePanel( std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink );
        ~ConsolePanel() override = default;

        void OnUIRender() override;

        bool IsVisible() const { return m_visible; }
        void SetVisible( bool v ) { m_visible = v; }
        void Clear() { m_clearTime = std::chrono::system_clock::now(); m_lastTotalCount = 0; }

    private:
        std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> m_sink;

        bool  m_visible    = true;
        bool  m_autoScroll = true;
        char  m_filter[ 256 ] = {};

        // Skip entries timestamped at-or-before m_clearTime (avoids stale index when buffer wraps).
        std::chrono::system_clock::time_point m_clearTime{};
        size_t                                m_lastTotalCount = 0;  // raw entry count for auto-scroll
        std::vector<size_t>                   m_visibleIndices;      // scratch, rebuilt each frame
    };
} // namespace Gaudi
