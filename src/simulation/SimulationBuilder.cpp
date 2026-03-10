#include "simulation/SimulationBuilder.h"

#include "core/Log.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#include "simulation/SimulationBlueprint.h"
#include <tuple>
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

        for( auto& field: gridFields )
        {
            if( field.textures[ 0 ].IsValid() )
                resourceManager->DestroyTexture( field.textures[ 0 ] );
            if( field.textures[ 1 ].IsValid() )
                resourceManager->DestroyTexture( field.textures[ 1 ] );
            if( field.interactionDeltaBuffer.IsValid() )
                resourceManager->DestroyBuffer( field.interactionDeltaBuffer );
        }
        gridFields.clear();

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
        const auto&     fields = blueprint.GetGridFields();

        if( groups.empty() && fields.empty() )
        {
            DT_WARN( "SimulationBuilder: Attempted to build an empty blueprint." );
            return state;
        }

        CompileGridFields( blueprint, state );

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
        if( !megaVertices.empty() )
            state.vertexBuffer =
                m_resourceManager->CreateBuffer( { megaVertices.size() * sizeof( Vertex ), BufferType::VERTEX, "SimulationVertexBuffer" } );

        if( !megaIndices.empty() )
            state.indexBuffer =
                m_resourceManager->CreateBuffer( { megaIndices.size() * sizeof( uint32_t ), BufferType::INDEX, "SimulationIndexBuffer" } );

        if( !indirectCommands.empty() )
            state.indirectCmdBuffer = m_resourceManager->CreateBuffer(
                { indirectCommands.size() * sizeof( VkDrawIndexedIndirectCommand ), BufferType::INDIRECT, "SimulationIndirectBuffer" } );

        if( !groupColors.empty() )
            state.groupDataBuffer = m_resourceManager->CreateBuffer(
                { groupColors.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "SimulationAdditionalDataBuffer" } );

        // Ping-pong agent buffers (Storage for Compute, Transfer Src/Dst for Readbacks and Uploads)
        BufferDesc agentDesc{ megaPositions.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "SimulationAgentBuffer" };
        state.agentBuffers[ 0 ] = m_resourceManager->CreateBuffer( agentDesc );
        state.agentBuffers[ 1 ] = m_resourceManager->CreateBuffer( agentDesc );

        // World size
        state.domainSize = blueprint.GetDomainSize();

        // 5. Upload Data to VRAM
        std::vector<BufferUploadRequest> uploads;

        if( state.vertexBuffer.IsValid() )
            uploads.push_back( { state.vertexBuffer, megaVertices.data(), megaVertices.size() * sizeof( Vertex ) } );

        if( state.indexBuffer.IsValid() )
            uploads.push_back( { state.indexBuffer, megaIndices.data(), megaIndices.size() * sizeof( uint32_t ) } );

        if( state.indirectCmdBuffer.IsValid() )
            uploads.push_back(
                { state.indirectCmdBuffer, indirectCommands.data(), indirectCommands.size() * sizeof( VkDrawIndexedIndirectCommand ) } );

        if( state.groupDataBuffer.IsValid() )
            uploads.push_back( { state.groupDataBuffer, groupColors.data(), groupColors.size() * sizeof( glm::vec4 ) } );

        // Always upload agents
        uploads.push_back( { state.agentBuffers[ 0 ], megaPositions.data(), megaPositions.size() * sizeof( glm::vec4 ) } );
        uploads.push_back( { state.agentBuffers[ 1 ], megaPositions.data(), megaPositions.size() * sizeof( glm::vec4 ) } );

        m_streamingManager->UploadBufferImmediate( uploads );

        CompileBehaviours( blueprint, state );

        DT_INFO( "SimulationBuilder: Successfully compiled Blueprint. Groups: {0}, Total Agents: {1}", validGroupCount, totalAgents );

        return state;
    }

    void SimulationBuilder::CompileGridFields( const SimulationBlueprint& blueprint, SimulationState& outState )
    {
        const auto& fields = blueprint.GetGridFields();
        if( fields.empty() )
            return;

        // Calculate grid dimensions based on physical domain and voxel resolution
        glm::vec3 domain    = blueprint.GetDomainSize();
        float     voxelSize = blueprint.GetVoxelSize();

        uint32_t gridWidth  = std::max( 1u, static_cast<uint32_t>( std::ceil( domain.x / voxelSize ) ) );
        uint32_t gridHeight = std::max( 1u, static_cast<uint32_t>( std::ceil( domain.y / voxelSize ) ) );
        uint32_t gridDepth  = std::max( 1u, static_cast<uint32_t>( std::ceil( domain.z / voxelSize ) ) );

        DT_INFO( "SimulationBuilder: Compiling {} Grid Fields at resolution {}x{}x{}", fields.size(), gridWidth, gridHeight, gridDepth );

        // Texture Descriptor for the PDE Grid
        // Note: Using R32_SFLOAT for now to simplify CPU-side array initialization.
        // We will optimize this to R16_SFLOAT in the future to halve VRAM bandwidth!
        TextureDesc texDesc{};
        texDesc.type   = TextureType::Texture3D;
        texDesc.format = VK_FORMAT_R32_SFLOAT;
        texDesc.width  = gridWidth;
        texDesc.height = gridHeight;
        texDesc.depth  = gridDepth;
        texDesc.usage  = TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST;

        ShaderHandle diffShader = m_resourceManager->CreateShader( "shaders/compute/diffusion.comp" );
        if( !diffShader.IsValid() )
        {
            DT_ASSERT( false, "" )
            DT_ERROR( "SimulationBuilder: CRITICAL! Failed to load 'diffusion.comp'. PDE solver will not work!" );
        }

        for( const auto& fieldDef: fields )
        {
            GridFieldState fieldState;
            fieldState.name             = fieldDef.GetName();
            fieldState.width            = gridWidth;
            fieldState.height           = gridHeight;
            fieldState.depth            = gridDepth;
            fieldState.currentReadIndex = 0;

            // Allocate Ping-Pong 3D Textures for PDE Solver
            texDesc.debugName        = "Texture_" + fieldDef.GetName();
            fieldState.textures[ 0 ] = m_resourceManager->CreateTexture( texDesc );
            fieldState.textures[ 1 ] = m_resourceManager->CreateTexture( texDesc );

            size_t     voxelCount = static_cast<size_t>( gridWidth ) * gridHeight * gridDepth;
            BufferDesc deltaDesc{ voxelCount * sizeof( int32_t ), BufferType::STORAGE, "DeltaBuffer_" + fieldDef.GetName() };
            fieldState.interactionDeltaBuffer = m_resourceManager->CreateBuffer( deltaDesc );

            // Initialize Grid Data (Upload from CPU to GPU)
            std::vector<int32_t> zeroDeltas( voxelCount, 0 );
            m_streamingManager->UploadBufferImmediate(
                { { fieldState.interactionDeltaBuffer, zeroDeltas.data(), zeroDeltas.size() * sizeof( int32_t ) } } );

            std::vector<float> initialData( voxelCount, 0.0f );

            // Domain bounds matching the shader (centered around 0,0,0)
            glm::vec3 boxMin = -domain * 0.5f;

            for( uint32_t z = 0; z < gridDepth; ++z )
            {
                for( uint32_t y = 0; y < gridHeight; ++y )
                {
                    for( uint32_t x = 0; x < gridWidth; ++x )
                    {
                        // Calculate the center of the current voxel in world space
                        glm::vec3 worldPos = boxMin + glm::vec3( x, y, z ) * voxelSize + glm::vec3( voxelSize * 0.5f );

                        // Call the user-defined lambda
                        float val = fieldDef.GetInitializer()( worldPos );

                        size_t index         = x + y * gridWidth + z * gridWidth * gridHeight;
                        initialData[ index ] = val;
                    }
                }
            }

            size_t byteSize = voxelCount * sizeof( float );

            // Synchronous upload of initial concentration to both ping-pong textures
            m_streamingManager->UploadTextureImmediate( fieldState.textures[ 0 ], initialData.data(), byteSize );
            m_streamingManager->UploadTextureImmediate( fieldState.textures[ 1 ], initialData.data(), byteSize );

            outState.gridFields.push_back( fieldState );

            ComputePipelineDesc pipeDesc{};
            pipeDesc.shader                  = diffShader;
            ComputePipelineHandle pipeHandle = m_resourceManager->CreatePipeline( pipeDesc );
            ComputePipeline*      pipe       = m_resourceManager->GetPipeline( pipeHandle );

            DT_ASSERT( pipeHandle.IsValid(), "" );

            // Binding Group 0: Read from Tex0, Write to Tex1
            BindingGroupHandle bgHandle0 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
            BindingGroup*      bg0       = m_resourceManager->GetBindingGroup( bgHandle0 );
            bg0->Bind( 0, m_resourceManager->GetTexture( fieldState.textures[ 0 ] ), VK_IMAGE_LAYOUT_GENERAL );
            bg0->Bind( 1, m_resourceManager->GetTexture( fieldState.textures[ 1 ] ), VK_IMAGE_LAYOUT_GENERAL );
            bg0->Bind( 2, m_resourceManager->GetBuffer( fieldState.interactionDeltaBuffer ) );
            bg0->Build();

            // Binding Group 1: Read from Tex1, Write to Tex0
            BindingGroupHandle bgHandle1 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
            BindingGroup*      bg1       = m_resourceManager->GetBindingGroup( bgHandle1 );
            bg1->Bind( 0, m_resourceManager->GetTexture( fieldState.textures[ 1 ] ), VK_IMAGE_LAYOUT_GENERAL );
            bg1->Bind( 1, m_resourceManager->GetTexture( fieldState.textures[ 0 ] ), VK_IMAGE_LAYOUT_GENERAL );
            bg1->Bind( 2, m_resourceManager->GetBuffer( fieldState.interactionDeltaBuffer ) );
            bg1->Build();

            ComputePushConstants pc{};
            pc.param1 = fieldDef.GetDiffusionCoefficient();
            pc.param2 = fieldDef.GetDecayRate();

            glm::uvec3 dispatchSize( ( gridWidth + 7 ) / 8, ( gridHeight + 7 ) / 8, ( gridDepth + 7 ) / 8 );
            outState.computeGraph.AddTask( ComputeTask( pipe, bg0, bg1, fieldDef.GetComputeHz(), pc, dispatchSize ) );
            DT_INFO( "SimulationBuilder: Compiled PDE task for '{}' at {}Hz", fieldDef.GetName(), fieldDef.GetComputeHz() );
        }
    }

    void SimulationBuilder::CompileBehaviours( const SimulationBlueprint& blueprint, SimulationState& outState )
    {
        uint32_t currentOffset = 0;

        for( const auto& group: blueprint.GetGroups() )
        {
            if( group.GetCount() == 0 )
                continue;

            for( const auto& record: group.GetBehaviours() )
            {
                if( std::holds_alternative<Behaviours::BrownianMotion>( record.behaviour ) )
                {
                    const auto&  brownian     = std::get<Behaviours::BrownianMotion>( record.behaviour );
                    ShaderHandle shaderHandle = m_resourceManager->CreateShader( "shaders/compute/brownian.comp" );

                    ComputePipelineDesc compDesc;
                    compDesc.shader                  = shaderHandle;
                    ComputePipelineHandle pipeHandle = m_resourceManager->CreatePipeline( compDesc );
                    ComputePipeline*      pipe       = m_resourceManager->GetPipeline( pipeHandle );

                    DT_ASSERT( pipeHandle.IsValid(), "" );

                    // 1. Create Binding Group 0 (Read from 0, Write to 1)
                    BindingGroupHandle bgHandle0 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                    BindingGroup*      bg0       = m_resourceManager->GetBindingGroup( bgHandle0 );
                    bg0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) ); // readonly
                    bg0->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) ); // writeonly
                    bg0->Build();

                    // 2. Create Binding Group 1 (Read from 1, Write to 0)
                    BindingGroupHandle bgHandle1 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                    BindingGroup*      bg1       = m_resourceManager->GetBindingGroup( bgHandle1 );
                    bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) ); // readonly
                    bg1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) ); // writeonly
                    bg1->Build();

                    ComputePushConstants pc{};
                    pc.param1 = brownian.speed;
                    pc.offset = currentOffset;
                    pc.count  = group.GetCount();

                    glm::uvec3 agentDispatch( ( group.GetCount() + 255 ) / 256, 1, 1 );
                    outState.computeGraph.AddTask( ComputeTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch ) );
                    DT_INFO( "SimulationBuilder: Compiled BrownianMotion for group '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::ConsumeField>( record.behaviour ) ||
                    std::holds_alternative<Behaviours::SecreteField>( record.behaviour ) )
                {
                    bool        isConsume  = std::holds_alternative<Behaviours::ConsumeField>( record.behaviour );
                    std::string targetName = isConsume ? std::get<Behaviours::ConsumeField>( record.behaviour ).fieldName
                                                       : std::get<Behaviours::SecreteField>( record.behaviour ).fieldName;

                    // Consume translates to negative rate, secrete to positive
                    float rate = isConsume ? -std::get<Behaviours::ConsumeField>( record.behaviour ).rate
                                           : std::get<Behaviours::SecreteField>( record.behaviour ).rate;

                    // Find target grid
                    GridFieldState* targetGrid = nullptr;
                    for( auto& grid: outState.gridFields )
                    {
                        if( grid.name == targetName )
                        {
                            targetGrid = &grid;
                            break;
                        }
                    }

                    if( targetGrid )
                    {
                        ShaderHandle        shaderHandle = m_resourceManager->CreateShader( "shaders/compute/field_interaction.comp" );
                        ComputePipelineDesc compDesc{};
                        compDesc.shader                  = shaderHandle;
                        ComputePipelineHandle pipeHandle = m_resourceManager->CreatePipeline( compDesc );
                        ComputePipeline*      pipe       = m_resourceManager->GetPipeline( pipeHandle );

                        // Read agents[0], write to DeltaBuffer
                        BindingGroupHandle bgHandle0 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                        BindingGroup*      bg0       = m_resourceManager->GetBindingGroup( bgHandle0 );
                        bg0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg0->Bind( 1, m_resourceManager->GetBuffer( targetGrid->interactionDeltaBuffer ) );
                        bg0->Build();

                        // Read agents[1], write to DeltaBuffer
                        BindingGroupHandle bgHandle1 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                        BindingGroup*      bg1       = m_resourceManager->GetBindingGroup( bgHandle1 );
                        bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Bind( 1, m_resourceManager->GetBuffer( targetGrid->interactionDeltaBuffer ) );
                        bg1->Build();

                        ComputePushConstants pc{};
                        pc.param1     = rate;
                        pc.offset     = currentOffset;
                        pc.count      = group.GetCount();
                        pc.domainSize = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                        pc.gridSize   = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth, 0 );

                        glm::uvec3 agentDispatch( ( group.GetCount() + 255 ) / 256, 1, 1 );
                        outState.computeGraph.AddTask( ComputeTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch ) );

                        DT_INFO( "SimulationBuilder: Compiled {} for '{}' at {}Hz", isConsume ? "Consumption" : "Secretion", targetName,
                                 record.targetHz );
                    }
                    else
                    {
                        DT_WARN( "SimulationBuilder: Target grid '{}' not found for agent interaction!", targetName );
                    }
                }
            }
            currentOffset += group.GetCount();
        }
    }

} // namespace DigitalTwin