#pragma once
#include <string>

namespace Gaudi
{
    class EditorPanel
    {
    public:
        virtual ~EditorPanel() = default;

        virtual void OnAttach() {}
        virtual void OnDetach() {}
        virtual void OnUpdate( float /* deltaTime */ ) {}
        virtual void OnUIRender() = 0;

        const std::string& GetName() const { return m_name; }

    protected:
        EditorPanel( const std::string& name )
            : m_name( name )
        {
        }

        std::string m_name;
    };
} // namespace Gaudi