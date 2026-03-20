#include "simulation/SimulationBuilder.h"

#include "core/Log.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationValidator.h"
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
        if( hashBuffer.IsValid() )
            resourceManager->DestroyBuffer( hashBuffer );
        if( offsetBuffer.IsValid() )
            resourceManager->DestroyBuffer( offsetBuffer );
        if( pressureBuffer.IsValid() )
            resourceManager->DestroyBuffer( pressureBuffer );
        if( agentCountBuffer.IsValid() )
            resourceManager->DestroyBuffer( agentCountBuffer );
        if( phenotypeBuffer.IsValid() )
            resourceManager->DestroyBuffer( phenotypeBuffer );
        if( signalingBuffer.IsValid() )
            resourceManager->DestroyBuffer( signalingBuffer );
        if( vesselEdgeBuffer.IsValid() )
            resourceManager->DestroyBuffer( vesselEdgeBuffer );
        if( vesselEdgeCountBuffer.IsValid() )
            resourceManager->DestroyBuffer( vesselEdgeCountBuffer );

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
        hashBuffer        = {};
        offsetBuffer      = {};
        pressureBuffer    = {};
        agentBuffers[ 0 ] = {};
        agentBuffers[ 1 ] = {};
        agentCountBuffer  = {};
        phenotypeBuffer       = {};
        signalingBuffer       = {};
        vesselEdgeBuffer      = {};
        vesselEdgeCountBuffer = {};
    }

    SimulationState SimulationBuilder::Build( const SimulationBlueprint& blueprint )
    {
        SimulationState state;

        // 0. Validate blueprint before touching any GPU resources
        ValidationResult validation = SimulationValidator::Validate( blueprint );
        for( const auto& issue : validation.issues )
        {
            if( issue.severity == ValidationIssue::Severity::Error )
                DT_ERROR( "[SimulationValidator] {}", issue.message );
            else
                DT_WARN( "[SimulationValidator] {}", issue.message );
        }
        if( !validation.IsValid() )
        {
            DT_ERROR( "[SimulationBuilder] Blueprint validation failed. Aborting build." );
            return state;
        }

        DT_INFO( "[SimulationBuilder] Starting simulation build process..." );

        // 1. Allocate GPU memory for agents and morphology (Mega-Buffers)
        AllocateAgentBuffers( blueprint, state );

        // 2. Build the Global Spatial Partitioning Grid (Hash, Sort, Build Offsets)
        // This is executed FIRST in the compute graph, ensuring all subsequent tasks
        // have an up-to-date spatial view of all agents.
        CompileSpatialGrid( blueprint, state );

        // 3. Compile PDEs and Fields (e.g., Oxygen, Glucose diffusion)
        CompileGridFields( blueprint, state );

        // 4. Compile specific cellular behaviours (Biomechanics, Chemotaxis, etc.)
        CompileBehaviours( blueprint, state );

        DT_INFO( "[SimulationBuilder] Simulation build process completed successfully." );

        return state;
    }

    void SimulationBuilder::AllocateAgentBuffers( const SimulationBlueprint& blueprint, SimulationState& outState )
    {
        const auto& groups = blueprint.GetGroups();

        // 1. Calculate required capacities for the Mega-Buffers
        uint32_t totalVertices   = 0;
        uint32_t totalIndices    = 0;
        uint32_t totalAgents     = 0;
        uint32_t validGroupCount = 0;

        std::vector<uint32_t> groupCapacities;
        for( const auto& group: groups )
        {
            if( group.GetCount() == 0 || group.GetPositions().empty() )
                continue;

            totalVertices += static_cast<uint32_t>( group.GetMorphology().vertices.size() );
            totalIndices += static_cast<uint32_t>( group.GetMorphology().indices.size() );

            uint32_t paddedCount = 131072;
            while( paddedCount < group.GetCount() )
                paddedCount <<= 1;

            groupCapacities.push_back( paddedCount );
            totalAgents += paddedCount;
            validGroupCount++;
        }

        if( totalAgents == 0 )
        {
            DT_WARN( "[SimulationBuilder] Blueprint contains 0 agents across all groups." );
            return;
        }

        outState.groupCount = validGroupCount;

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
        uint32_t groupIdx              = 0;

        for( const auto& group: groups )
        {
            if( group.GetCount() == 0 || group.GetPositions().empty() )
                continue;

            uint32_t    capacity = groupCapacities[ groupIdx++ ];
            const auto& morph    = group.GetMorphology();

            // Append Geometry
            megaVertices.insert( megaVertices.end(), morph.vertices.begin(), morph.vertices.end() );
            megaIndices.insert( megaIndices.end(), morph.indices.begin(), morph.indices.end() );

            // Append Initial State (Positions)
            uint32_t copyCount = std::min( group.GetCount(), static_cast<uint32_t>( group.GetPositions().size() ) );
            megaPositions.insert( megaPositions.end(), group.GetPositions().begin(), group.GetPositions().begin() + copyCount );

            // Pad with zeros if the user provided fewer positions than the requested count
            for( uint32_t i = copyCount; i < capacity; ++i )
            {
                megaPositions.push_back( glm::vec4( 0.0f, 0.0f, 0.0f, 0.0f ) );
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
            currentInstanceOffset += capacity;
        }

        // 4. Create core GPU Buffers via ResourceManager
        if( !megaVertices.empty() )
            outState.vertexBuffer =
                m_resourceManager->CreateBuffer( { megaVertices.size() * sizeof( Vertex ), BufferType::VERTEX, "SimulationVertexBuffer" } );

        if( !megaIndices.empty() )
            outState.indexBuffer =
                m_resourceManager->CreateBuffer( { megaIndices.size() * sizeof( uint32_t ), BufferType::INDEX, "SimulationIndexBuffer" } );

        if( !indirectCommands.empty() )
            outState.indirectCmdBuffer = m_resourceManager->CreateBuffer(
                { indirectCommands.size() * sizeof( VkDrawIndexedIndirectCommand ), BufferType::INDIRECT, "SimulationIndirectBuffer" } );

        if( !groupColors.empty() )
            outState.groupDataBuffer = m_resourceManager->CreateBuffer(
                { groupColors.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "SimulationAdditionalDataBuffer" } );

        // Ping-pong agent buffers
        BufferDesc agentDesc{ megaPositions.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "SimulationAgentBuffer" };
        outState.agentBuffers[ 0 ] = m_resourceManager->CreateBuffer( agentDesc );
        outState.agentBuffers[ 1 ] = m_resourceManager->CreateBuffer( agentDesc );
        outState.agentCountBuffer =
            m_resourceManager->CreateBuffer( { outState.groupCount * sizeof( uint32_t ), BufferType::INDIRECT, "AgentCountBuffer" } );

        // 5. Upload Data to VRAM synchronously
        std::vector<BufferUploadRequest> uploads;

        if( outState.vertexBuffer.IsValid() )
            uploads.push_back( { outState.vertexBuffer, megaVertices.data(), megaVertices.size() * sizeof( Vertex ) } );

        if( outState.indexBuffer.IsValid() )
            uploads.push_back( { outState.indexBuffer, megaIndices.data(), megaIndices.size() * sizeof( uint32_t ) } );

        if( outState.indirectCmdBuffer.IsValid() )
            uploads.push_back(
                { outState.indirectCmdBuffer, indirectCommands.data(), indirectCommands.size() * sizeof( VkDrawIndexedIndirectCommand ) } );

        if( outState.groupDataBuffer.IsValid() )
            uploads.push_back( { outState.groupDataBuffer, groupColors.data(), groupColors.size() * sizeof( glm::vec4 ) } );

        // Always upload agents
        uploads.push_back( { outState.agentBuffers[ 0 ], megaPositions.data(), megaPositions.size() * sizeof( glm::vec4 ) } );
        uploads.push_back( { outState.agentBuffers[ 1 ], megaPositions.data(), megaPositions.size() * sizeof( glm::vec4 ) } );
        std::vector<uint32_t> initialCounts;
        for( const auto& group: blueprint.GetGroups() )
        {
            initialCounts.push_back( group.GetCount() );
        }
        uploads.push_back( { outState.agentCountBuffer, initialCounts.data(), initialCounts.size() * sizeof( uint32_t ), 0 } );

        m_streamingManager->UploadBufferImmediate( uploads );

        DT_INFO( "[SimulationBuilder] Agent Mega-Buffers allocated. Total Agents: {}", totalAgents );
    }

    void SimulationBuilder::CompileSpatialGrid( const SimulationBlueprint& blueprint, SimulationState& outState )
    {
        // 1. Calculate total agents across all groups for the global grid
        uint32_t totalAgents = 0;
        for( const auto& group: blueprint.GetGroups() )
        {
            totalAgents += group.GetCount();
        }

        if( totalAgents == 0 )
            return;

        // 2. Fetch Spatial Configuration from Blueprint
        const auto& spatialConfig = blueprint.GetSpatialPartitioning();
        float       computeHz     = spatialConfig.computeHz;
        float       cellSize      = spatialConfig.cellSize;

        // GPU Bitonic Sort requires the array size to be a power of two
        // Use the global maximum capacity for all spatial data structures!
        uint32_t maxCapacity     = 131072;
        uint32_t paddedCount     = maxCapacity; // 131072 is already a perfect power of two
        uint32_t offsetArraySize = 262144;      // 64x64x64 hash grid slots

        // 3. Allocate Spatial & Physics Buffers (if not already allocated)
        if( !outState.hashBuffer.IsValid() )
        {
            outState.hashBuffer = m_resourceManager->CreateBuffer( { paddedCount * 8, BufferType::STORAGE, "AgentHashes" } );
        }
        if( !outState.offsetBuffer.IsValid() )
        {
            outState.offsetBuffer = m_resourceManager->CreateBuffer( { offsetArraySize * 4, BufferType::STORAGE, "CellOffsets" } );

            // Clear Offsets with 0xFFFFFFFF (Empty Cell Marker) using StreamingManager
            std::vector<uint32_t>            emptyOffsets( offsetArraySize, 0xFFFFFFFF );
            std::vector<BufferUploadRequest> uploads = { { outState.offsetBuffer, emptyOffsets.data(), offsetArraySize * 4, 0 } };
            m_streamingManager->UploadBufferImmediate( uploads );
        }
        if( !outState.pressureBuffer.IsValid() )
        {
            // We keep pressure buffer here since it's a global physics property
            // shared across potentially multiple mechanical behaviours.
            outState.pressureBuffer = m_resourceManager->CreateBuffer( { paddedCount * 4, BufferType::STORAGE, "AgentPressures" } );
        }

        // 4. Load Shaders and Create Pipelines for Spatial Partitioning
        ComputePipelineDesc hashDesc{};
        hashDesc.shader                      = m_resourceManager->CreateShader( "shaders/compute/hash_agents.comp" );
        hashDesc.debugName                   = "HashAgentsPipeline";
        ComputePipelineHandle hashPipeHandle = m_resourceManager->CreatePipeline( hashDesc );
        ComputePipeline*      hashPipe       = m_resourceManager->GetPipeline( hashPipeHandle );

        ComputePipelineDesc sortDesc{};
        sortDesc.shader                      = m_resourceManager->CreateShader( "shaders/compute/bitonic_sort.comp" );
        sortDesc.debugName                   = "BitonicSortPipeline";
        ComputePipelineHandle sortPipeHandle = m_resourceManager->CreatePipeline( sortDesc );
        ComputePipeline*      sortPipe       = m_resourceManager->GetPipeline( sortPipeHandle );

        ComputePipelineDesc offsetDesc{};
        offsetDesc.shader                      = m_resourceManager->CreateShader( "shaders/compute/build_offsets.comp" );
        offsetDesc.debugName                   = "BuildOffsetsPipeline";
        ComputePipelineHandle offsetPipeHandle = m_resourceManager->CreatePipeline( offsetDesc );
        ComputePipeline*      offsetPipe       = m_resourceManager->GetPipeline( offsetPipeHandle );

        // 5. Setup Binding Groups
        BindingGroup* bgHash0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( hashPipeHandle, 0 ) );
        bgHash0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
        bgHash0->Bind( 1, m_resourceManager->GetBuffer( outState.hashBuffer ) );
        bgHash0->Build();

        BindingGroup* bgHash1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( hashPipeHandle, 0 ) );
        bgHash1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
        bgHash1->Bind( 1, m_resourceManager->GetBuffer( outState.hashBuffer ) );
        bgHash1->Build();

        BindingGroup* bgSort = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( sortPipeHandle, 0 ) );
        bgSort->Bind( 0, m_resourceManager->GetBuffer( outState.hashBuffer ) );
        bgSort->Build();

        BindingGroup* bgOffset = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( offsetPipeHandle, 0 ) );
        bgOffset->Bind( 0, m_resourceManager->GetBuffer( outState.hashBuffer ) );
        bgOffset->Bind( 1, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
        bgOffset->Build();

        // 6. Base Push Constants
        ComputePushConstants basePC{};
        basePC.offset      = 0; // Processing all agents globally, offset is always 0
        basePC.maxCapacity = paddedCount;
        // w component acts as padding here since maxRadius is not needed for pure hashing
        basePC.domainSize = glm::vec4( blueprint.GetDomainSize(), 0.0f );
        basePC.gridSize   = glm::uvec4( 0 );

        // --- TASK A: HASH AGENTS ---
        ComputePushConstants hashPC = basePC;
        hashPC.fParam0              = cellSize;
        glm::uvec3 hashDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
        outState.computeGraph.AddTask( ComputeTask( hashPipe, bgHash0, bgHash1, computeHz, hashPC, hashDispatch ) );

        // --- TASK B: BITONIC SORT (The Magic GPU Loop) ---
        glm::uvec3 sortDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
        for( uint32_t k = 2; k <= paddedCount; k <<= 1 )
        {
            for( uint32_t j = k >> 1; j > 0; j >>= 1 )
            {
                ComputePushConstants sortPC = basePC;
                sortPC.maxCapacity          = paddedCount; // Sort operates on the padded size
                sortPC.uParam0              = j;
                sortPC.uParam1              = k;
                outState.computeGraph.AddTask( ComputeTask( sortPipe, bgSort, bgSort, computeHz, sortPC, sortDispatch ) );
            }
        }

        // --- TASK C: BUILD OFFSETS ---
        ComputePushConstants offsetPC = basePC;
        offsetPC.maxCapacity          = paddedCount;
        glm::uvec3 offsetDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
        outState.computeGraph.AddTask( ComputeTask( offsetPipe, bgOffset, bgOffset, computeHz, offsetPC, offsetDispatch ) );

        DT_INFO( "[SimulationBuilder] Compiled Global Spatial Grid. Total Agents: {}, Frequency: {}Hz", paddedCount, computeHz );
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
        texDesc.usage  = TextureUsage::STORAGE | TextureUsage::SAMPLED | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST;

        ShaderHandle diffShader = m_resourceManager->CreateShader( "shaders/compute/diffusion.comp" );
        if( !diffShader.IsValid() )
        {
            DT_ASSERT( false, "" )
            DT_ERROR( "SimulationBuilder: CRITICAL! Failed to load 'diffusion.comp'. PDE solver will not work!" );
        }

        uint32_t fieldIndex = 0;
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
            pc.fParam0 = fieldDef.GetDiffusionCoefficient();
            pc.fParam1 = fieldDef.GetDecayRate();

            glm::uvec3  dispatchSize( ( gridWidth + 7 ) / 8, ( gridHeight + 7 ) / 8, ( gridDepth + 7 ) / 8 );
            ComputeTask diffTask( pipe, bg0, bg1, fieldDef.GetComputeHz(), pc, dispatchSize );
            diffTask.SetTag( "diffusion_" + std::to_string( fieldIndex ) );
            outState.computeGraph.AddTask( diffTask );
            DT_INFO( "SimulationBuilder: Compiled PDE task for '{}' at {}Hz", fieldDef.GetName(), fieldDef.GetComputeHz() );
            fieldIndex++;
        }
    }

    void SimulationBuilder::CompileBehaviours( const SimulationBlueprint& blueprint, SimulationState& outState )
    {
        uint32_t currentOffset = 0;
        uint32_t groupIndex    = 0;

        // Base Push Constants common to almost all agent behaviours
        ComputePushConstants basePC{};
        basePC.domainSize = glm::vec4( blueprint.GetDomainSize(), 0.0f ); // Radius will be injected later if needed
        basePC.gridSize   = glm::uvec4( 0 );                              // Default to no grid interaction

        // Pre-fetch the Oxygen Grid resolution IF it exists, otherwise use 0s
        if( !outState.gridFields.empty() )
        {
            // outState.gridFields is a vector of SimulationState::GridFieldState (which HAS width/height/depth)
            // We use the GPU-compiled state size, not the CPU blueprint!
            basePC.gridSize = glm::uvec4( outState.gridFields[ 0 ].width, outState.gridFields[ 0 ].height, outState.gridFields[ 0 ].depth, 0 );
        }

        // Maintain the global capacity to match memory structures
        uint32_t paddedCount     = 131072;
        uint32_t offsetArraySize = 262144;

        uint32_t behaviourIndex = 0;

        for( const auto& group: blueprint.GetGroups() )
        {
            if( group.GetCount() == 0 )
                continue;

            behaviourIndex = 0;
            for( const auto& record: group.GetBehaviours() )
            {
                std::string tagBase = "_" + std::to_string( groupIndex ) + "_" + std::to_string( behaviourIndex );

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
                    bg0->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    if( outState.phenotypeBuffer.IsValid() )
                        bg0->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    else
                        bg0->Bind( 3, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bg0->Build();

                    // 2. Create Binding Group 1 (Read from 1, Write to 0)
                    BindingGroupHandle bgHandle1 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                    BindingGroup*      bg1       = m_resourceManager->GetBindingGroup( bgHandle1 );
                    bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) ); // readonly
                    bg1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) ); // writeonly
                    bg1->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    if( outState.phenotypeBuffer.IsValid() )
                        bg1->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    else
                        bg1->Bind( 3, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bg1->Build();

                    ComputePushConstants pc{};
                    pc.fParam0     = brownian.speed;
                    pc.fParam1     = static_cast<float>( record.requiredCellType );
                    pc.offset      = currentOffset;
                    pc.maxCapacity = paddedCount;
                    pc.uParam0     = static_cast<uint32_t>( record.requiredLifecycleState ); // -1 → 0xFFFFFFFF
                    pc.uParam1     = groupIndex;

                    glm::uvec3 agentDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask brownianTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch );
                    brownianTask.SetTag( "brownian" + tagBase );
                    brownianTask.SetChainFlip( true );
                    outState.computeGraph.AddTask( brownianTask );
                    DT_INFO( "SimulationBuilder: Compiled BrownianMotion for group '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::Biomechanics>( record.behaviour ) )
                {
                    auto& biomechanics = std::get<Behaviours::Biomechanics>( record.behaviour );

                    uint32_t offsetArraySize = 262144; // 64x64x64 hash grid slots (must match CompileSpatialGrid)

                    // 1. Load Shaders and Create Pipeline for Biomechanics
                    ComputePipelineDesc jkrDesc{};
                    jkrDesc.shader                      = m_resourceManager->CreateShader( "shaders/compute/jkr_forces.comp" );
                    jkrDesc.debugName                   = "JKRForcesPipeline";
                    ComputePipelineHandle jkrPipeHandle = m_resourceManager->CreatePipeline( jkrDesc );
                    ComputePipeline*      jkrPipe       = m_resourceManager->GetPipeline( jkrPipeHandle );

                    // 2. Setup Binding Groups
                    // Note: hashBuffer, offsetBuffer, and pressureBuffer are globally built in CompileSpatialGrid
                    BindingGroup* bgJkr0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( jkrPipeHandle, 0 ) );
                    bgJkr0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgJkr0->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgJkr0->Bind( 2, m_resourceManager->GetBuffer( outState.pressureBuffer ) );
                    bgJkr0->Bind( 3, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                    bgJkr0->Bind( 4, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                    bgJkr0->Bind( 5, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    if( outState.phenotypeBuffer.IsValid() )
                        bgJkr0->Bind( 6, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    else
                        bgJkr0->Bind( 6, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgJkr0->Build();

                    BindingGroup* bgJkr1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( jkrPipeHandle, 0 ) );
                    bgJkr1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgJkr1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgJkr1->Bind( 2, m_resourceManager->GetBuffer( outState.pressureBuffer ) );
                    bgJkr1->Bind( 3, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                    bgJkr1->Bind( 4, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                    bgJkr1->Bind( 5, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    if( outState.phenotypeBuffer.IsValid() )
                        bgJkr1->Bind( 6, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    else
                        bgJkr1->Bind( 6, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgJkr1->Build();

                    // 3. Configure Task specific parameters
                    ComputePushConstants jkrPC = basePC;
                    jkrPC.offset               = currentOffset;
                    jkrPC.maxCapacity          = paddedCount;
                    jkrPC.fParam0              = biomechanics.repulsionStiffness;
                    jkrPC.fParam1              = biomechanics.adhesionStrength;
                    jkrPC.fParam2              = static_cast<float>( record.requiredLifecycleState );
                    jkrPC.fParam3              = static_cast<float>( record.requiredCellType );
                    jkrPC.uParam0              = offsetArraySize;
                    jkrPC.uParam1              = groupIndex;
                    jkrPC.domainSize           = glm::vec4( blueprint.GetDomainSize(), biomechanics.maxRadius );

                    // 4. Append Task to Compute Graph
                    glm::uvec3 jkrDispatch( ( paddedCount + 255 ) / 256, 1, 1 );

                    ComputeTask jkrTask( jkrPipe, bgJkr0, bgJkr1, record.targetHz, jkrPC, jkrDispatch );
                    jkrTask.SetTag( "jkr" + tagBase );
                    jkrTask.SetChainFlip( true );
                    outState.computeGraph.AddTask( jkrTask );

                    DT_INFO( "[SimulationBuilder] Compiled Biomechanics (JKR) for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::CellCycle>( record.behaviour ) )
                {
                    const auto& cellCycle = std::get<Behaviours::CellCycle>( record.behaviour );

                    // 1. Allocate Phenotype Buffer if it doesn't exist
                    if( !outState.phenotypeBuffer.IsValid() )
                    {
                        uint32_t globalCapacity = 0;
                        for( const auto& g: blueprint.GetGroups() )
                        {
                            uint32_t tc = g.GetCount() * 2;
                            if( tc < 64 )
                                tc = 64;
                            uint32_t pc = 1;
                            while( pc < tc )
                                pc <<= 1;
                            globalCapacity += pc;
                        }

                        size_t phenotypeSize     = globalCapacity * 4 * sizeof( uint32_t );
                        outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );

                        struct PhenotypeData
                        {
                            uint32_t lifecycleState;
                            float    biomass;
                            float    timer;
                            uint32_t cellType;
                        };
                        std::vector<PhenotypeData> initialPhenotypes( paddedCount, { 0, 0.5f, 0.0f, 0 } );

                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initialPhenotypes.data(), phenotypeSize, 0 } } );
                    }

                    // Pressure buffer is absolutely required for Contact Inhibition
                    if( !outState.pressureBuffer.IsValid() )
                    {
                        uint32_t globalCapacity = 0;
                        for( const auto& g: blueprint.GetGroups() )
                        {
                            uint32_t tc = g.GetCount() * 2;
                            if( tc < 64 )
                                tc = 64;
                            uint32_t pc = 1;
                            while( pc < tc )
                                pc <<= 1;
                            globalCapacity += pc;
                        }

                        size_t pressureSize     = globalCapacity * sizeof( float );
                        outState.pressureBuffer = m_resourceManager->CreateBuffer( { pressureSize, BufferType::STORAGE, "PressureBuffer" } );

                        std::vector<float> initialPressures( globalCapacity, 0.0f );
                        m_streamingManager->UploadBufferImmediate( { { outState.pressureBuffer, initialPressures.data(), pressureSize, 0 } } );
                    }

                    // 2. Create Pipelines
                    ComputePipelineDesc   phenoDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/update_phenotype.comp" ),
                                                   "UpdatePhenotype" };
                    ComputePipelineHandle phenoPipeHandle = m_resourceManager->CreatePipeline( phenoDesc );
                    ComputePipeline*      phenoPipe       = m_resourceManager->GetPipeline( phenoPipeHandle );

                    ComputePipelineDesc   mitosisDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/mitosis_append.comp" ),
                                                     "MitosisAppend" };
                    ComputePipelineHandle mitosisPipeHandle = m_resourceManager->CreatePipeline( mitosisDesc );
                    ComputePipeline*      mitosisPipe       = m_resourceManager->GetPipeline( mitosisPipeHandle );

                    // 3. Create Binding Groups (Update Phenotype)
                    BindingGroup* bgPheno0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( phenoPipeHandle, 0 ) );
                    bgPheno0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgPheno0->Bind( 1, m_resourceManager->GetBuffer( outState.pressureBuffer ) );
                    // Fallback to satisfy Vulkan Validation Layers if Oxygen Grid is missing
                    if( !outState.gridFields.empty() && outState.gridFields[ 0 ].textures[ 0 ].IsValid() )
                        bgPheno0->Bind( 2, m_resourceManager->GetTexture( outState.gridFields[ 0 ].textures[ 0 ] ), VK_IMAGE_LAYOUT_GENERAL );
                    bgPheno0->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgPheno0->Bind( 4, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgPheno0->Build();

                    BindingGroup* bgPheno1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( phenoPipeHandle, 0 ) );
                    bgPheno1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgPheno1->Bind( 1, m_resourceManager->GetBuffer( outState.pressureBuffer ) );
                    if( !outState.gridFields.empty() && outState.gridFields[ 0 ].textures[ 0 ].IsValid() )
                        bgPheno1->Bind( 2, m_resourceManager->GetTexture( outState.gridFields[ 0 ].textures[ 0 ] ), VK_IMAGE_LAYOUT_GENERAL );
                    bgPheno1->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgPheno1->Bind( 4, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgPheno1->Build();

                    // Create Binding Groups (Mitosis)
                    BindingGroup* bgMitosis0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( mitosisPipeHandle, 0 ) );
                    bgMitosis0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgMitosis0->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgMitosis0->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgMitosis0->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgMitosis0->Build();

                    BindingGroup* bgMitosis1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( mitosisPipeHandle, 0 ) );
                    bgMitosis1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgMitosis1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgMitosis1->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgMitosis1->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgMitosis1->Build();

                    // 4. Configure Push Constants
                    ComputePushConstants phenoPC = basePC;
                    phenoPC.offset               = currentOffset;
                    phenoPC.maxCapacity          = paddedCount;
                    phenoPC.fParam0              = cellCycle.growthRatePerSec;
                    phenoPC.fParam1              = cellCycle.targetO2;
                    phenoPC.fParam2              = cellCycle.arrestPressure;
                    phenoPC.fParam3              = cellCycle.necrosisO2;
                    phenoPC.fParam4              = cellCycle.hypoxiaO2;
                    phenoPC.fParam5              = cellCycle.apoptosisProbPerSec;
                    phenoPC.uParam1              = groupIndex;

                    ComputePushConstants mitosisPC = basePC;
                    mitosisPC.offset               = currentOffset;
                    mitosisPC.maxCapacity          = paddedCount;
                    mitosisPC.uParam1              = groupIndex;

                    // 5. Append Tasks to Compute Graph
                    glm::uvec3 maxDispatch( ( paddedCount + 255 ) / 256, 1, 1 );

                    ComputeTask phenoTask( phenoPipe, bgPheno0, bgPheno1, record.targetHz, phenoPC, maxDispatch );
                    phenoTask.SetTag( "phenotype" + tagBase );
                    outState.computeGraph.AddTask( phenoTask );

                    ComputeTask mitosisTask( mitosisPipe, bgMitosis0, bgMitosis1, record.targetHz, mitosisPC, maxDispatch );
                    mitosisTask.SetTag( "mitosis" + tagBase );
                    mitosisTask.SetChainFlip( true );
                    outState.computeGraph.AddTask( mitosisTask );

                    DT_INFO( "[SimulationBuilder] Compiled Biology (Cell Cycle) for '{}' at {}Hz", group.GetName(), record.targetHz );
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

                    // Require lifecycle state
                    int requiredLifecycleState = isConsume
                                                     ? std::get<Behaviours::ConsumeField>( record.behaviour ).requiredLifecycleState
                                                     : std::get<Behaviours::SecreteField>( record.behaviour ).requiredLifecycleState;

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
                        if( requiredLifecycleState != -1 && !outState.phenotypeBuffer.IsValid() )
                        {
                            uint32_t globalCapacity = 0;
                            for( const auto& g: blueprint.GetGroups() )
                            {
                                uint32_t tc = g.GetCount() * 2;
                                if( tc < 64 )
                                    tc = 64;
                                uint32_t pc = 1;
                                while( pc < tc )
                                    pc <<= 1;
                                globalCapacity += pc;
                            }
                            size_t phenotypeSize     = globalCapacity * 4 * sizeof( uint32_t );
                            outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );
                            struct PhenotypeData
                            {
                                uint32_t lifecycleState;
                                float    biomass;
                                float    timer;
                                uint32_t cellType;
                            };
                            std::vector<PhenotypeData> initialPhenotypes( globalCapacity, { 0, 0.5f, 0.0f, 0 } );
                            m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initialPhenotypes.data(), phenotypeSize, 0 } } );
                        }

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
                        bg0->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        if( outState.phenotypeBuffer.IsValid() )
                            bg0->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        else
                            bg0->Bind( 3, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg0->Build();

                        // Read agents[1], write to DeltaBuffer
                        BindingGroupHandle bgHandle1 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                        BindingGroup*      bg1       = m_resourceManager->GetBindingGroup( bgHandle1 );
                        bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Bind( 1, m_resourceManager->GetBuffer( targetGrid->interactionDeltaBuffer ) );
                        bg1->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        if( outState.phenotypeBuffer.IsValid() )
                            bg1->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        else
                            bg1->Bind( 3, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Build();

                        ComputePushConstants pc{};
                        pc.fParam0     = rate;
                        pc.fParam1     = static_cast<float>( requiredLifecycleState );
                        pc.offset      = currentOffset;
                        pc.maxCapacity = paddedCount;
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth, 0 );

                        glm::uvec3 agentDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask chemTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch );
                        chemTask.SetTag( "chemfield" + tagBase );
                        outState.computeGraph.AddTask( chemTask );

                        DT_INFO( "SimulationBuilder: Compiled {} for '{}' at {}Hz", isConsume ? "Consumption" : "Secretion", targetName,
                                 record.targetHz );
                    }
                    else
                    {
                        DT_WARN( "SimulationBuilder: Target grid '{}' not found for agent interaction!", targetName );
                    }
                }
                if( std::holds_alternative<Behaviours::Chemotaxis>( record.behaviour ) )
                {
                    const auto& chemo = std::get<Behaviours::Chemotaxis>( record.behaviour );

                    GridFieldState* targetGrid = nullptr;
                    for( auto& grid: outState.gridFields )
                        if( grid.name == chemo.fieldName ) { targetGrid = &grid; break; }

                    if( targetGrid )
                    {
                        ShaderHandle          sh       = m_resourceManager->CreateShader( "shaders/compute/chemotaxis.comp" );
                        ComputePipelineDesc   desc{};
                        desc.shader                    = sh;
                        ComputePipelineHandle pipe     = m_resourceManager->CreatePipeline( desc );
                        ComputePipeline*      pipePtr  = m_resourceManager->GetPipeline( pipe );

                        BindingGroup* bg0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( pipe, 0 ) );
                        bg0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg0->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg0->Bind( 2, m_resourceManager->GetTexture( targetGrid->textures[ 0 ] ), VK_IMAGE_LAYOUT_GENERAL );
                        bg0->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        if( outState.phenotypeBuffer.IsValid() )
                            bg0->Bind( 4, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        else
                            bg0->Bind( 4, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg0->Build();

                        BindingGroup* bg1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( pipe, 0 ) );
                        bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg1->Bind( 2, m_resourceManager->GetTexture( targetGrid->textures[ 1 ] ), VK_IMAGE_LAYOUT_GENERAL );
                        bg1->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        if( outState.phenotypeBuffer.IsValid() )
                            bg1->Bind( 4, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        else
                            bg1->Bind( 4, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Build();

                        ComputePushConstants pc{};
                        pc.fParam0     = chemo.chemotacticSensitivity;
                        pc.fParam1     = chemo.receptorSaturation;
                        pc.fParam2     = chemo.maxVelocity;
                        pc.fParam3     = static_cast<float>( record.requiredCellType );
                        pc.offset      = currentOffset;
                        pc.maxCapacity = paddedCount;
                        pc.uParam0     = static_cast<uint32_t>( record.requiredLifecycleState ); // -1 → 0xFFFFFFFF
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth, 0 );

                        glm::uvec3  dispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask task( pipePtr, bg0, bg1, record.targetHz, pc, dispatch );
                        task.SetTag( "chemotaxis" + tagBase );
                        task.SetChainFlip( true );
                        outState.computeGraph.AddTask( task );
                        DT_INFO( "[SimulationBuilder] Compiled Chemotaxis for '{}' → '{}' at {}Hz",
                                 group.GetName(), chemo.fieldName, record.targetHz );
                    }
                    else
                    {
                        DT_WARN( "[SimulationBuilder] Chemotaxis: target field '{}' not found!", chemo.fieldName );
                    }
                }
                if( std::holds_alternative<Behaviours::NotchDll4>( record.behaviour ) )
                {
                    const auto& notch = std::get<Behaviours::NotchDll4>( record.behaviour );

                    // Allocate signaling buffer once (global, covers all agent slots)
                    if( !outState.signalingBuffer.IsValid() )
                    {
                        uint32_t globalCapacity = 0;
                        for( const auto& g: blueprint.GetGroups() )
                        {
                            if( g.GetCount() == 0 )
                                continue;
                            uint32_t cap = 131072;
                            while( cap < g.GetCount() )
                                cap <<= 1;
                            globalCapacity += cap;
                        }

                        struct SignalingData
                        {
                            float dll4;
                            float nicd;
                            float vegfr2;
                            float pad;
                        };
                        size_t signalingSize    = globalCapacity * sizeof( SignalingData );
                        outState.signalingBuffer = m_resourceManager->CreateBuffer( { signalingSize, BufferType::STORAGE, "SignalingBuffer" } );

                        // Initial state: dll4=0.5 (mid-range), vegfr2=1.0 (fully expressed)
                        std::vector<SignalingData> initSignaling( globalCapacity, { 0.5f, 0.0f, 1.0f, 0.0f } );
                        m_streamingManager->UploadBufferImmediate( { { outState.signalingBuffer, initSignaling.data(), signalingSize, 0 } } );
                    }

                    // Allocate phenotype buffer if no CellCycle behaviour did so
                    if( !outState.phenotypeBuffer.IsValid() )
                    {
                        uint32_t globalCapacity = 0;
                        for( const auto& g: blueprint.GetGroups() )
                        {
                            if( g.GetCount() == 0 )
                                continue;
                            uint32_t cap = 131072;
                            while( cap < g.GetCount() )
                                cap <<= 1;
                            globalCapacity += cap;
                        }

                        struct PhenotypeData
                        {
                            uint32_t lifecycleState;
                            float    biomass;
                            float    timer;
                            uint32_t cellType;
                        };
                        size_t phenotypeSize     = globalCapacity * sizeof( PhenotypeData );
                        outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );

                        std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0u, 0.5f, 0.0f, 0u } );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                    }

                    // Pipeline
                    ComputePipelineDesc   notchDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/notch_dll4.comp" ), "NotchDll4Pipeline" };
                    ComputePipelineHandle notchPipeHandle = m_resourceManager->CreatePipeline( notchDesc );
                    ComputePipeline*      notchPipe       = m_resourceManager->GetPipeline( notchPipeHandle );

                    // Binding groups — ping-pong on agent read buffer; signaling/phenotype are single shared buffers
                    BindingGroup* bgNotch0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( notchPipeHandle, 0 ) );
                    bgNotch0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgNotch0->Bind( 1, m_resourceManager->GetBuffer( outState.signalingBuffer ) );
                    bgNotch0->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgNotch0->Bind( 3, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                    bgNotch0->Bind( 4, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                    bgNotch0->Bind( 5, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgNotch0->Build();

                    BindingGroup* bgNotch1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( notchPipeHandle, 0 ) );
                    bgNotch1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgNotch1->Bind( 1, m_resourceManager->GetBuffer( outState.signalingBuffer ) );
                    bgNotch1->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgNotch1->Bind( 3, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                    bgNotch1->Bind( 4, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                    bgNotch1->Bind( 5, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgNotch1->Build();

                    // Push constants
                    float signalingRadius = blueprint.GetSpatialPartitioning().cellSize * 0.5f;

                    ComputePushConstants notchPC{};
                    notchPC.fParam0     = notch.dll4ProductionRate;
                    notchPC.fParam1     = notch.dll4DecayRate;
                    notchPC.fParam2     = notch.notchInhibitionGain;
                    notchPC.fParam3     = notch.vegfr2BaseExpression;
                    notchPC.fParam4     = notch.tipThreshold;
                    notchPC.fParam5     = notch.stalkThreshold;
                    notchPC.offset      = currentOffset;
                    notchPC.maxCapacity = paddedCount;
                    notchPC.uParam0     = offsetArraySize;
                    notchPC.uParam1     = groupIndex;
                    notchPC.domainSize  = glm::vec4( blueprint.GetDomainSize(), signalingRadius );

                    glm::uvec3  notchDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask notchTask( notchPipe, bgNotch0, bgNotch1, record.targetHz, notchPC, notchDispatch );
                    notchTask.SetTag( "notch" + tagBase );
                    outState.computeGraph.AddTask( notchTask );

                    DT_INFO( "[SimulationBuilder] Compiled NotchDll4 for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::Anastomosis>( record.behaviour ) )
                {
                    const auto& anastomosis = std::get<Behaviours::Anastomosis>( record.behaviour );

                    // Ensure phenotype buffer exists (stores cellType)
                    if( !outState.phenotypeBuffer.IsValid() )
                    {
                        uint32_t globalCapacity = 0;
                        for( const auto& g: blueprint.GetGroups() )
                        {
                            if( g.GetCount() == 0 )
                                continue;
                            uint32_t cap = 131072;
                            while( cap < g.GetCount() )
                                cap <<= 1;
                            globalCapacity += cap;
                        }

                        struct PhenotypeData
                        {
                            uint32_t lifecycleState;
                            float    biomass;
                            float    timer;
                            uint32_t cellType;
                        };
                        size_t phenotypeSize     = globalCapacity * sizeof( PhenotypeData );
                        outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );

                        std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0u, 0.5f, 0.0f, 0u } );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                    }

                    // Allocate vessel edge buffers (once per simulation — accumulate across frames)
                    if( !outState.vesselEdgeBuffer.IsValid() )
                    {
                        struct VesselEdge
                        {
                            uint32_t agentA;
                            uint32_t agentB;
                            float    dist;
                            uint32_t flags;
                        };
                        size_t edgeBufferSize    = paddedCount * sizeof( VesselEdge );
                        outState.vesselEdgeBuffer = m_resourceManager->CreateBuffer( { edgeBufferSize, BufferType::STORAGE, "VesselEdgeBuffer" } );

                        outState.vesselEdgeCountBuffer =
                            m_resourceManager->CreateBuffer( { sizeof( uint32_t ), BufferType::STORAGE, "VesselEdgeCountBuffer" } );
                        uint32_t zero = 0;
                        m_streamingManager->UploadBufferImmediate( { { outState.vesselEdgeCountBuffer, &zero, sizeof( uint32_t ) } } );
                    }

                    ComputePipelineDesc   anastomosisDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/anastomosis.comp" ),
                                                          "AnastomosisPipeline" };
                    ComputePipelineHandle anastomosisPipeHandle = m_resourceManager->CreatePipeline( anastomosisDesc );
                    ComputePipeline*      anastomosisPipe       = m_resourceManager->GetPipeline( anastomosisPipeHandle );

                    BindingGroup* bgAnastomosis0 =
                        m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( anastomosisPipeHandle, 0 ) );
                    bgAnastomosis0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgAnastomosis0->Bind( 1, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgAnastomosis0->Bind( 2, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                    bgAnastomosis0->Bind( 3, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                    bgAnastomosis0->Bind( 4, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgAnastomosis0->Bind( 5, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgAnastomosis0->Bind( 6, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgAnastomosis0->Build();

                    BindingGroup* bgAnastomosis1 =
                        m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( anastomosisPipeHandle, 0 ) );
                    bgAnastomosis1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgAnastomosis1->Bind( 1, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgAnastomosis1->Bind( 2, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                    bgAnastomosis1->Bind( 3, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                    bgAnastomosis1->Bind( 4, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgAnastomosis1->Bind( 5, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgAnastomosis1->Bind( 6, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgAnastomosis1->Build();

                    float hashCellSize = blueprint.GetSpatialPartitioning().cellSize;

                    ComputePushConstants anastomosisPC{};
                    anastomosisPC.fParam0     = anastomosis.contactDistance;
                    anastomosisPC.offset      = currentOffset;
                    anastomosisPC.maxCapacity = paddedCount;
                    anastomosisPC.uParam0     = offsetArraySize;
                    anastomosisPC.uParam1     = groupIndex;
                    anastomosisPC.domainSize  = glm::vec4( blueprint.GetDomainSize(), hashCellSize );

                    glm::uvec3  anastomosisDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask anastomosisTask( anastomosisPipe, bgAnastomosis0, bgAnastomosis1, record.targetHz, anastomosisPC,
                                                anastomosisDispatch );
                    anastomosisTask.SetTag( "anastomosis" + tagBase );
                    // No ChainFlip — reads positions only, writes phenotype and edge buffers
                    outState.computeGraph.AddTask( anastomosisTask );

                    DT_INFO( "[SimulationBuilder] Compiled Anastomosis for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                behaviourIndex++;
            }
            currentOffset += paddedCount;
            groupIndex++;
        }
    }

    void SimulationBuilder::UpdateParameters( const SimulationBlueprint& blueprint, SimulationState& state )
    {
        // ── Grid Fields ───────────────────────────────────────────────────────
        const auto& fields = blueprint.GetGridFields();
        for( uint32_t i = 0; i < static_cast<uint32_t>( fields.size() ); ++i )
        {
            ComputeTask* task = state.computeGraph.FindTask( "diffusion_" + std::to_string( i ) );
            if( task )
            {
                ComputePushConstants pc = task->GetPushConstants();
                pc.fParam0              = fields[ i ].GetDiffusionCoefficient();
                pc.fParam1              = fields[ i ].GetDecayRate();
                task->UpdatePushConstants( pc );
            }
        }

        // ── Agent Behaviours ──────────────────────────────────────────────────
        uint32_t groupIndex = 0;

        for( const auto& group: blueprint.GetGroups() )
        {
            if( group.GetCount() == 0 )
                continue;

            uint32_t behaviourIndex = 0;
            for( const auto& record: group.GetBehaviours() )
            {
                std::string tagBase = "_" + std::to_string( groupIndex ) + "_" + std::to_string( behaviourIndex );

                if( std::holds_alternative<Behaviours::BrownianMotion>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "brownian" + tagBase );
                    if( task )
                    {
                        ComputePushConstants pc    = task->GetPushConstants();
                        pc.fParam0                 = std::get<Behaviours::BrownianMotion>( record.behaviour ).speed;
                        pc.fParam1                 = static_cast<float>( record.requiredCellType );
                        pc.uParam0                 = static_cast<uint32_t>( record.requiredLifecycleState );
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::Biomechanics>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "jkr" + tagBase );
                    if( task )
                    {
                        const auto&          bio = std::get<Behaviours::Biomechanics>( record.behaviour );
                        ComputePushConstants pc   = task->GetPushConstants();
                        pc.fParam0                = bio.repulsionStiffness;
                        pc.fParam1                = bio.adhesionStrength;
                        pc.fParam2                = static_cast<float>( record.requiredLifecycleState );
                        pc.fParam3                = static_cast<float>( record.requiredCellType );
                        pc.domainSize.w           = bio.maxRadius; // w holds maxRadius for JKR
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::CellCycle>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "phenotype" + tagBase );
                    if( task )
                    {
                        const auto&          cc = std::get<Behaviours::CellCycle>( record.behaviour );
                        ComputePushConstants pc  = task->GetPushConstants();
                        pc.fParam0               = cc.growthRatePerSec;
                        pc.fParam1               = cc.targetO2;
                        pc.fParam2               = cc.arrestPressure;
                        pc.fParam3               = cc.necrosisO2;
                        pc.fParam4               = cc.hypoxiaO2;
                        pc.fParam5               = cc.apoptosisProbPerSec;
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::ConsumeField>( record.behaviour ) ||
                         std::holds_alternative<Behaviours::SecreteField>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "chemfield" + tagBase );
                    if( task )
                    {
                        bool  isConsume         = std::holds_alternative<Behaviours::ConsumeField>( record.behaviour );
                        float rate              = isConsume ? -std::get<Behaviours::ConsumeField>( record.behaviour ).rate
                                                            : std::get<Behaviours::SecreteField>( record.behaviour ).rate;
                        int   reqLifecycleState = isConsume
                                                      ? std::get<Behaviours::ConsumeField>( record.behaviour ).requiredLifecycleState
                                                      : std::get<Behaviours::SecreteField>( record.behaviour ).requiredLifecycleState;

                        ComputePushConstants pc = task->GetPushConstants();
                        pc.fParam0              = rate;
                        pc.fParam1              = static_cast<float>( reqLifecycleState );
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::Chemotaxis>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "chemotaxis" + tagBase );
                    if( task )
                    {
                        const auto&          chemo = std::get<Behaviours::Chemotaxis>( record.behaviour );
                        ComputePushConstants pc    = task->GetPushConstants();
                        pc.fParam0                 = chemo.chemotacticSensitivity;
                        pc.fParam1                 = chemo.receptorSaturation;
                        pc.fParam2                 = chemo.maxVelocity;
                        pc.fParam3                 = static_cast<float>( record.requiredCellType );
                        pc.uParam0                 = static_cast<uint32_t>( record.requiredLifecycleState );
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::NotchDll4>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "notch" + tagBase );
                    if( task )
                    {
                        const auto&          notch = std::get<Behaviours::NotchDll4>( record.behaviour );
                        ComputePushConstants pc    = task->GetPushConstants();
                        pc.fParam0                 = notch.dll4ProductionRate;
                        pc.fParam1                 = notch.dll4DecayRate;
                        pc.fParam2                 = notch.notchInhibitionGain;
                        pc.fParam3                 = notch.vegfr2BaseExpression;
                        pc.fParam4                 = notch.tipThreshold;
                        pc.fParam5                 = notch.stalkThreshold;
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::Anastomosis>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "anastomosis" + tagBase );
                    if( task )
                    {
                        const auto&          anastomosis = std::get<Behaviours::Anastomosis>( record.behaviour );
                        ComputePushConstants pc          = task->GetPushConstants();
                        pc.fParam0                       = anastomosis.contactDistance;
                        task->UpdatePushConstants( pc );
                    }
                }

                behaviourIndex++;
            }
            groupIndex++;
        }
    }

} // namespace DigitalTwin