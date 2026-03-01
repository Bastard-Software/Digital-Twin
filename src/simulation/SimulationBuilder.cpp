#include "simulation/SimulationBuilder.h"

#include "core/Log.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "simulation/SimulationBlueprint.h"
#include <volk.h>

namespace DigitalTwin
{
    SimulationBuilder::SimulationBuilder( ResourceManager* resourceManager, StreamingManager* streamingManager )
        : m_resourceManager( resourceManager )
        , m_streamingManager( streamingManager )
    {
    }

    void SimulationState::Destroy( ResourceManager* resourceManager )
    {
        if( !resourceManager )
            return;

        if( vertexBuffer.IsValid() )
            resourceManager->DestroyBuffer( vertexBuffer );
        if( indexBuffer.IsValid() )
            resourceManager->DestroyBuffer( indexBuffer );
        if( indirectCmdBuffer.IsValid() )
            resourceManager->DestroyBuffer( indirectCmdBuffer );
        if( groupDataBuffer.IsValid() )
            resourceManager->DestroyBuffer( groupDataBuffer );
        if( agentBuffers[ 0 ].IsValid() )
            resourceManager->DestroyBuffer( agentBuffers[ 0 ] );
        if( agentBuffers[ 1 ].IsValid() )
            resourceManager->DestroyBuffer( agentBuffers[ 1 ] );

        vertexBuffer      = {};
        indexBuffer       = {};
        indirectCmdBuffer = {};
        groupDataBuffer   = {};
        agentBuffers[ 0 ] = {};
        agentBuffers[ 1 ] = {};
    }

    SimulationState SimulationBuilder::Build( const SimulationBlueprint& blueprint )
    {
        SimulationState state{};
        const auto&     groups = blueprint.GetGroups();

        if( groups.empty() )
        {
            DT_WARN( "SimulationBuilder: Attempted to build an empty blueprint." );
            return state;
        }

        // 1. Calculate required capacities for the Mega-Buffers
        uint32_t totalVertices   = 0;
        uint32_t totalIndices    = 0;
        uint32_t totalAgents     = 0;
        uint32_t validGroupCount = 0;

        for( const auto& group: groups )
        {
            if( group.GetCount() == 0 || group.GetPositions().empty() )
                continue;

            totalVertices += static_cast<uint32_t>( group.GetMorphology().vertices.size() );
            totalIndices += static_cast<uint32_t>( group.GetMorphology().indices.size() );
            totalAgents += group.GetCount();
            validGroupCount++;
        }

        if( totalAgents == 0 )
        {
            DT_WARN( "SimulationBuilder: Blueprint contains 0 agents across all groups." );
            return state;
        }

        state.groupCount = validGroupCount;

        // 2. Allocate CPU-side continuous memory blocks
        std::vector<Vertex>                       megaVertices;
        std::vector<uint32_t>                     megaIndices;
        std::vector<glm::vec4>                    megaPositions;
        std::vector<VkDrawIndexedIndirectCommand> indirectCommands;
        std::vector<glm::vec4>                    groupColors;

        megaVertices.reserve( totalVertices );
        megaIndices.reserve( totalIndices );
        megaPositions.reserve( totalAgents );
        indirectCommands.reserve( validGroupCount );
        groupColors.reserve( validGroupCount );

        // 3. Compile data and calculate offsets
        uint32_t currentVertexOffset   = 0;
        uint32_t currentIndexOffset    = 0;
        uint32_t currentInstanceOffset = 0;

        for( const auto& group: groups )
        {
            if( group.GetCount() == 0 || group.GetPositions().empty() )
                continue;

            const auto& morph = group.GetMorphology();

            // Append Geometry
            megaVertices.insert( megaVertices.end(), morph.vertices.begin(), morph.vertices.end() );
            megaIndices.insert( megaIndices.end(), morph.indices.begin(), morph.indices.end() );

            // Append Initial State (Positions)
            // Ensure we don't copy more positions than the requested count, or fewer if mismatched
            uint32_t copyCount = std::min( group.GetCount(), static_cast<uint32_t>( group.GetPositions().size() ) );
            megaPositions.insert( megaPositions.end(), group.GetPositions().begin(), group.GetPositions().begin() + copyCount );

            // Pad with zeros if the user provided fewer positions than the requested count
            for( uint32_t i = copyCount; i < group.GetCount(); ++i )
            {
                megaPositions.push_back( glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );
            }

            // Create Indirect Draw Command for this specific group
            VkDrawIndexedIndirectCommand cmd{};
            cmd.indexCount    = static_cast<uint32_t>( morph.indices.size() );
            cmd.instanceCount = group.GetCount();
            cmd.firstIndex    = currentIndexOffset;
            cmd.vertexOffset  = currentVertexOffset;
            cmd.firstInstance = currentInstanceOffset;

            indirectCommands.push_back( cmd );
            groupColors.push_back( group.GetColor() );

            // Advance offsets for the next group
            currentVertexOffset += static_cast<uint32_t>( morph.vertices.size() );
            currentIndexOffset += static_cast<uint32_t>( morph.indices.size() );
            currentInstanceOffset += group.GetCount();
        }

        // 4. Create GPU Buffers via ResourceManager
        state.vertexBuffer = m_resourceManager->CreateBuffer( { megaVertices.size() * sizeof( Vertex ), BufferType::VERTEX } );
        state.indexBuffer  = m_resourceManager->CreateBuffer( { megaIndices.size() * sizeof( uint32_t ), BufferType::INDEX } );
        state.indirectCmdBuffer =
            m_resourceManager->CreateBuffer( { indirectCommands.size() * sizeof( VkDrawIndexedIndirectCommand ), BufferType::INDIRECT } );
        state.groupDataBuffer = m_resourceManager->CreateBuffer( { groupColors.size() * sizeof( glm::vec4 ), BufferType::STORAGE } );

        // Ping-pong agent buffers (Storage for Compute, Transfer Src/Dst for Readbacks and Uploads)
        BufferDesc agentDesc{ megaPositions.size() * sizeof( glm::vec4 ), BufferType::STORAGE };
        state.agentBuffers[ 0 ] = m_resourceManager->CreateBuffer( agentDesc );
        state.agentBuffers[ 1 ] = m_resourceManager->CreateBuffer( agentDesc );

        // 5. Upload Data to VRAM
        m_streamingManager->UploadBufferImmediate( state.vertexBuffer, megaVertices.data(), megaVertices.size() * sizeof( Vertex ) );
        m_streamingManager->UploadBufferImmediate( state.indexBuffer, megaIndices.data(), megaIndices.size() * sizeof( uint32_t ) );
        m_streamingManager->UploadBufferImmediate( state.indirectCmdBuffer, indirectCommands.data(),
                                                   indirectCommands.size() * sizeof( VkDrawIndexedIndirectCommand ) );
        m_streamingManager->UploadBufferImmediate( state.groupDataBuffer, groupColors.data(), groupColors.size() * sizeof( glm::vec4 ) );

        // Initialize both ping-pong buffers with the starting positions
        m_streamingManager->UploadBufferImmediate( state.agentBuffers[ 0 ], megaPositions.data(), megaPositions.size() * sizeof( glm::vec4 ) );
        m_streamingManager->UploadBufferImmediate( state.agentBuffers[ 1 ], megaPositions.data(), megaPositions.size() * sizeof( glm::vec4 ) );

        DT_INFO( "SimulationBuilder: Successfully compiled Blueprint. Groups: {0}, Total Agents: {1}", validGroupCount, totalAgents );

        return state;
    }

} // namespace DigitalTwin