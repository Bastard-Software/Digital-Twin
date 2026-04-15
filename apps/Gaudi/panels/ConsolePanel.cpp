#include "ConsolePanel.h"

#include <imgui.h>
#include <spdlog/details/log_msg_buffer.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>

namespace Gaudi
{
    ConsolePanel::ConsolePanel( std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink )
        : EditorPanel( "Console" )
        , m_sink( std::move( sink ) )
    {
    }

    void ConsolePanel::OnUIRender()
    {
        if( !m_visible )
            return;

        if( !ImGui::Begin( m_name.c_str(), &m_visible ) )
        {
            ImGui::End();
            return;
        }

        // ── Toolbar ──────────────────────────────────────────────────────────
        if( ImGui::Button( "Clear" ) )
            Clear();
        ImGui::SameLine();
        ImGui::Checkbox( "Auto-scroll", &m_autoScroll );
        ImGui::SameLine();
        ImGui::SetNextItemWidth( 200.0f );
        ImGui::InputText( "Filter", m_filter, sizeof( m_filter ) );

        ImGui::Separator();

        // ── Log body ─────────────────────────────────────────────────────────
        const float footerHeight = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();
        ImGui::BeginChild( "##consolelog", ImVec2( 0, -footerHeight ), false,
                           ImGuiWindowFlags_HorizontalScrollbar );

        const auto entries = m_sink->last_raw();

        // Auto-scroll: trigger whenever the raw buffer gains new entries.
        const bool hasNew  = entries.size() != m_lastTotalCount;
        m_lastTotalCount   = entries.size();

        // Case-insensitive filter: pre-compute lowercase copy once per frame.
        const char* filterRaw = m_filter[ 0 ] ? m_filter : nullptr;
        char        filterLower[ 256 ] = {};
        if( filterRaw )
        {
            for( int i = 0; m_filter[ i ]; ++i )
                filterLower[ i ] = static_cast<char>( tolower( static_cast<unsigned char>( m_filter[ i ] ) ) );
        }

        // Build visible index (skip cleared entries and filtered-out entries).
        // Timestamp comparison correctly handles ring-buffer wraps — no index arithmetic needed.
        m_visibleIndices.clear();
        m_visibleIndices.reserve( entries.size() );
        for( size_t i = 0; i < entries.size(); ++i )
        {
            const auto& msg = entries[ i ];
            if( msg.time <= m_clearTime )
                continue;

            if( filterRaw )
            {
                // Build lowercase "logger: payload" string on the stack for matching.
                char lineBuf[ 1024 ];
                int  len = snprintf( lineBuf, sizeof( lineBuf ), "%.*s: %.*s",
                                     (int)msg.logger_name.size(), msg.logger_name.data(),
                                     (int)msg.payload.size(),     msg.payload.data() );
                for( int j = 0; j < len; ++j )
                    lineBuf[ j ] = static_cast<char>( tolower( static_cast<unsigned char>( lineBuf[ j ] ) ) );
                if( !strstr( lineBuf, filterLower ) )
                    continue;
            }

            m_visibleIndices.push_back( i );
        }

        // Virtualised rendering: only process rows actually on screen.
        ImGuiListClipper clipper;
        clipper.Begin( static_cast<int>( m_visibleIndices.size() ) );
        while( clipper.Step() )
        {
            for( int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row )
            {
                const auto& msg = entries[ m_visibleIndices[ static_cast<size_t>( row ) ] ];

                // Format timestamp HH:MM:SS
                const auto tt = std::chrono::system_clock::to_time_t( msg.time );
                struct tm  tmBuf{};
#ifdef _WIN32
                localtime_s( &tmBuf, &tt );
#else
                localtime_r( &tt, &tmBuf );
#endif
                char timeBuf[ 12 ];
                snprintf( timeBuf, sizeof( timeBuf ), "%02d:%02d:%02d",
                          tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec );

                // Colour by level
                ImVec4 col;
                switch( msg.level )
                {
                case spdlog::level::trace:    col = ImVec4( 0.60f, 0.60f, 0.60f, 1.0f ); break;
                case spdlog::level::debug:    col = ImVec4( 0.75f, 0.75f, 0.75f, 1.0f ); break;
                case spdlog::level::info:     col = ImVec4( 1.00f, 1.00f, 1.00f, 1.0f ); break;
                case spdlog::level::warn:     col = ImVec4( 1.00f, 0.85f, 0.20f, 1.0f ); break;
                case spdlog::level::err:      col = ImVec4( 1.00f, 0.35f, 0.35f, 1.0f ); break;
                case spdlog::level::critical: col = ImVec4( 1.00f, 0.20f, 0.20f, 1.0f ); break;
                default:                      col = ImVec4( 0.80f, 0.80f, 0.80f, 1.0f ); break;
                }

                // No heap allocation: %.*s reads directly from spdlog's string_view data.
                ImGui::TextColored( col, "%s  %.*s: %.*s", timeBuf,
                                    (int)msg.logger_name.size(), msg.logger_name.data(),
                                    (int)msg.payload.size(),     msg.payload.data() );
            }
        }
        clipper.End();

        if( m_autoScroll && hasNew )
            ImGui::SetScrollHereY( 1.0f );

        ImGui::EndChild();

        ImGui::End();
    }

} // namespace Gaudi
