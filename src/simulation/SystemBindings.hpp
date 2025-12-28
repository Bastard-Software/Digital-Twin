#pragma once
#include "rhi/BindingGroup.hpp"
#include "simulation/SimulationContext.hpp"
#include <array>
#include <string>

namespace DigitalTwin
{
    /**
     * @brief Helper class for managing Double-Buffered Descriptor Sets.
     * Used inside the OnConfigureSystems builder lambda.
     */
    class SystemBindings
    {
    public:
        SystemBindings( SimulationContext* context, Ref<ComputeKernel> kernel )
            : m_context( context )
        {
            m_groups[ 0 ] = kernel->CreateBindingGroup();
            m_groups[ 1 ] = kernel->CreateBindingGroup();
        }

        void SetUniform( const std::string& name, Ref<Buffer> buffer )
        {
            if( m_groups[ 0 ] )
                m_groups[ 0 ]->Set( name, buffer );
            if( m_groups[ 1 ] )
                m_groups[ 1 ]->Set( name, buffer );
        }

        void SetInput( const std::string& name )
        {
            // Input: Frame 0 reads Buf 0, Frame 1 reads Buf 1
            m_groups[ 0 ]->Set( name, m_context->GetBuffer( 0 ) );
            m_groups[ 1 ]->Set( name, m_context->GetBuffer( 1 ) );
        }

        void SetOutput( const std::string& name )
        {
            // Output: Frame 0 writes Buf 1, Frame 1 writes Buf 0
            m_groups[ 0 ]->Set( name, m_context->GetBuffer( 1 ) );
            m_groups[ 1 ]->Set( name, m_context->GetBuffer( 0 ) );
        }

        void Build()
        {
            if( m_groups[ 0 ] )
                m_groups[ 0 ]->Build();
            if( m_groups[ 1 ] )
                m_groups[ 1 ]->Build();
        }

        /**
         * @brief Returns the raw BindingGroup for the specific frame index.
         * This allows ComputeGraph to remain unaware of SystemBindings.
         */
        Ref<BindingGroup> Get( uint32_t frameIndex ) const { return m_groups[ frameIndex % 2 ]; }

    private:
        SimulationContext*               m_context;
        std::array<Ref<BindingGroup>, 2> m_groups;
    };
} // namespace DigitalTwin