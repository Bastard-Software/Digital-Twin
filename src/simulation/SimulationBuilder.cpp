#include "simulation/SimulationBuilder.h"

#include "core/Log.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationValidator.h"
#include <numeric>
#include <random>
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
        if( orientationBuffer.IsValid() )
            resourceManager->DestroyBuffer( orientationBuffer );
        if( signalingBuffer.IsValid() )
            resourceManager->DestroyBuffer( signalingBuffer );
        if( vesselEdgeBuffer.IsValid() )
            resourceManager->DestroyBuffer( vesselEdgeBuffer );
        if( vesselEdgeCountBuffer.IsValid() )
            resourceManager->DestroyBuffer( vesselEdgeCountBuffer );
        if( vesselComponentBuffer.IsValid() )
            resourceManager->DestroyBuffer( vesselComponentBuffer );
        if( agentReorderBuffer.IsValid() )
            resourceManager->DestroyBuffer( agentReorderBuffer );
        if( drawMetaBuffer.IsValid() )
            resourceManager->DestroyBuffer( drawMetaBuffer );

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
        vesselEdgeBuffer       = {};
        vesselEdgeCountBuffer  = {};
        vesselComponentBuffer  = {};
        agentReorderBuffer     = {};
        drawMetaBuffer         = {};
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

        state.domainSize = blueprint.GetDomainSize();

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

        // DrawMeta: one per draw command, tells build_indirect.comp which group/cellType each command targets
        struct DrawMeta
        {
            uint32_t groupIndex;
            uint32_t targetCellType; // 0xFFFFFFFF = default (any unmatched cellType)
            uint32_t groupOffset;    // first agent slot for this group in agent buffer
            uint32_t groupCapacity;  // padded capacity of this group
        };

        // 1. Calculate required capacities for the Mega-Buffers
        uint32_t totalVertices      = 0;
        uint32_t totalIndices       = 0;
        uint32_t totalAgents        = 0;
        uint32_t totalReorderSlots  = 0;
        uint32_t totalDrawCommands  = 0;
        uint32_t validGroupCount    = 0;

        std::vector<uint32_t> groupCapacities;
        for( const auto& group: groups )
        {
            if( group.GetCount() == 0 || group.GetPositions().empty() )
                continue;

            // Default mesh
            totalVertices += static_cast<uint32_t>( group.GetMorphology().vertices.size() );
            totalIndices  += static_cast<uint32_t>( group.GetMorphology().indices.size() );

            // Cell-type morphology meshes
            for( const auto& entry: group.GetCellTypeMorphologies() )
            {
                totalVertices += static_cast<uint32_t>( entry.mesh.vertices.size() );
                totalIndices  += static_cast<uint32_t>( entry.mesh.indices.size() );
            }

            uint32_t paddedCount = 131072;
            while( paddedCount < group.GetCount() )
                paddedCount <<= 1;

            groupCapacities.push_back( paddedCount );
            totalAgents += paddedCount;

            // Draw commands: 1 (default) + N (cell-type morphologies)
            uint32_t drawsForGroup = 1 + static_cast<uint32_t>( group.GetCellTypeMorphologies().size() );
            totalDrawCommands += drawsForGroup;

            // Reorder slots: each draw command gets capacity slots (worst case all agents have same cellType)
            totalReorderSlots += drawsForGroup * paddedCount;

            validGroupCount++;
        }

        if( totalAgents == 0 )
        {
            DT_WARN( "[SimulationBuilder] Blueprint contains 0 agents across all groups." );
            return;
        }

        outState.groupCount       = validGroupCount;
        outState.drawCommandCount = totalDrawCommands;
        outState.totalPaddedAgents = totalAgents;

        // 2. Allocate CPU-side continuous memory blocks
        std::vector<Vertex>                       megaVertices;
        std::vector<uint32_t>                     megaIndices;
        std::vector<glm::vec4>                    megaPositions;
        std::vector<glm::vec4>                    megaOrientations; // per-agent normals; default (0,1,0,0) when not provided
        std::vector<VkDrawIndexedIndirectCommand> indirectCommands;
        std::vector<glm::vec4>                    groupColors;
        std::vector<DrawMeta>                     drawMetaEntries;

        megaVertices.reserve( totalVertices );
        megaIndices.reserve( totalIndices );
        megaPositions.reserve( totalAgents );
        megaOrientations.reserve( totalAgents );
        indirectCommands.reserve( totalDrawCommands );
        groupColors.reserve( totalDrawCommands );
        drawMetaEntries.reserve( totalDrawCommands );

        // 3. Compile data and calculate offsets
        uint32_t currentVertexOffset   = 0;
        uint32_t currentIndexOffset    = 0;
        uint32_t currentAgentOffset    = 0; // offset into agent position buffer
        uint32_t currentReorderOffset  = 0; // offset into reorder buffer
        uint32_t groupIdx              = 0;

        for( const auto& group: groups )
        {
            if( group.GetCount() == 0 || group.GetPositions().empty() )
                continue;

            uint32_t    capacity       = groupCapacities[ groupIdx ];
            const auto& morph          = group.GetMorphology();
            const auto& cellTypeMorphs = group.GetCellTypeMorphologies();

            // --- Default mesh (catches any cellType not in cellTypeMorphs) ---
            megaVertices.insert( megaVertices.end(), morph.vertices.begin(), morph.vertices.end() );
            megaIndices.insert( megaIndices.end(), morph.indices.begin(), morph.indices.end() );

            VkDrawIndexedIndirectCommand defaultCmd{};
            defaultCmd.indexCount    = static_cast<uint32_t>( morph.indices.size() );
            defaultCmd.instanceCount = 0; // filled by build_indirect.comp
            defaultCmd.firstIndex    = currentIndexOffset;
            defaultCmd.vertexOffset  = currentVertexOffset;
            defaultCmd.firstInstance  = currentReorderOffset;
            indirectCommands.push_back( defaultCmd );
            groupColors.push_back( group.GetColor() );
            drawMetaEntries.push_back( { groupIdx, 0xFFFFFFFF, currentAgentOffset, capacity } );

            currentVertexOffset += static_cast<uint32_t>( morph.vertices.size() );
            currentIndexOffset  += static_cast<uint32_t>( morph.indices.size() );
            currentReorderOffset += capacity;

            // --- Cell-type-specific meshes ---
            for( const auto& entry: cellTypeMorphs )
            {
                megaVertices.insert( megaVertices.end(), entry.mesh.vertices.begin(), entry.mesh.vertices.end() );
                megaIndices.insert( megaIndices.end(), entry.mesh.indices.begin(), entry.mesh.indices.end() );

                VkDrawIndexedIndirectCommand ctCmd{};
                ctCmd.indexCount    = static_cast<uint32_t>( entry.mesh.indices.size() );
                ctCmd.instanceCount = 0; // filled by build_indirect.comp
                ctCmd.firstIndex    = currentIndexOffset;
                ctCmd.vertexOffset  = currentVertexOffset;
                ctCmd.firstInstance  = currentReorderOffset;
                indirectCommands.push_back( ctCmd );
                groupColors.push_back( entry.color.x >= 0.0f ? entry.color : group.GetColor() );
                drawMetaEntries.push_back( { groupIdx, static_cast<uint32_t>( entry.cellTypeIndex ), currentAgentOffset, capacity } );

                currentVertexOffset  += static_cast<uint32_t>( entry.mesh.vertices.size() );
                currentIndexOffset   += static_cast<uint32_t>( entry.mesh.indices.size() );
                currentReorderOffset += capacity;
            }

            // --- Agent positions ---
            uint32_t copyCount = std::min( group.GetCount(), static_cast<uint32_t>( group.GetPositions().size() ) );
            megaPositions.insert( megaPositions.end(), group.GetPositions().begin(), group.GetPositions().begin() + copyCount );
            for( uint32_t i = copyCount; i < capacity; ++i )
                megaPositions.push_back( glm::vec4( 0.0f ) );

            // --- Agent orientations (per-cell outward normal; default +Y when not provided) ---
            {
                const auto& orientations = group.GetOrientations();
                uint32_t    oriCount     = std::min( group.GetCount(), static_cast<uint32_t>( orientations.size() ) );
                megaOrientations.insert( megaOrientations.end(), orientations.begin(), orientations.begin() + oriCount );
                for( uint32_t i = oriCount; i < capacity; ++i )
                    megaOrientations.push_back( glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f ) ); // default: face +Y
            }

            currentAgentOffset += capacity;
            groupIdx++;
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

        // Orientation buffer — static, single copy, written once at init
        outState.orientationBuffer = m_resourceManager->CreateBuffer(
            { megaOrientations.size() * sizeof( glm::vec4 ), BufferType::STORAGE, "OrientationBuffer" } );

        // Reorder buffer: maps draw-command instance indices → global agent indices
        outState.agentReorderBuffer = m_resourceManager->CreateBuffer(
            { totalReorderSlots * sizeof( uint32_t ), BufferType::STORAGE, "AgentReorderBuffer" } );

        // Draw meta buffer: per-draw-command metadata for build_indirect.comp
        outState.drawMetaBuffer = m_resourceManager->CreateBuffer(
            { drawMetaEntries.size() * sizeof( DrawMeta ), BufferType::STORAGE, "DrawMetaBuffer" } );

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
        uploads.push_back( { outState.orientationBuffer, megaOrientations.data(), megaOrientations.size() * sizeof( glm::vec4 ) } );
        std::vector<uint32_t> initialCounts;
        for( const auto& group: blueprint.GetGroups() )
        {
            initialCounts.push_back( group.GetCount() );
        }
        uploads.push_back( { outState.agentCountBuffer, initialCounts.data(), initialCounts.size() * sizeof( uint32_t ), 0 } );

        // Upload reorder buffer (identity init not needed — shader fills it every frame)
        // Upload draw meta
        uploads.push_back( { outState.drawMetaBuffer, drawMetaEntries.data(), drawMetaEntries.size() * sizeof( DrawMeta ) } );

        m_streamingManager->UploadBufferImmediate( uploads );

        DT_INFO( "[SimulationBuilder] Agent Mega-Buffers allocated. Total Agents: {}, Draw Commands: {}", totalAgents, totalDrawCommands );
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

        // Pre-pass: if any group specifies a non-zero initial cell type, create the phenotype buffer
        // now with the correct per-group cellType values. All lazy-init checks below will then skip.
        {
            bool anyNonDefault = false;
            for( const auto& g: blueprint.GetGroups() )
                if( g.GetCount() > 0 && g.GetInitialCellType() != 0 )
                    anyNonDefault = true;

            if( anyNonDefault && !outState.phenotypeBuffer.IsValid() )
            {
                struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };

                uint32_t globalCapacity = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;
                    globalCapacity += cap;
                }

                size_t phenotypeSize     = globalCapacity * sizeof( PhenotypeData );
                outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );
                std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0u, 0.5f, 0.0f, 0u } );

                // Apply per-group initial cell types
                uint32_t off = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;
                    if( g.GetInitialCellType() != 0 )
                    {
                        for( uint32_t i = 0; i < cap; i++ )
                            initPhenotypes[ off + i ].cellType = static_cast<uint32_t>( g.GetInitialCellType() );
                    }
                    off += cap;
                }

                m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
            }
        }

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

                    // Brownian always needs the phenotype buffer (dead-cell guard)
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
                        size_t phenotypeSize     = globalCapacity * sizeof( uint32_t ) * 4;
                        outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );
                        struct PhenotypeData
                        {
                            uint32_t lifecycleState;
                            float    biomass;
                            float    timer;
                            uint32_t cellType;
                        };
                        std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0, 0.5f, 0.0f, 0 } );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                    }

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
                    bg0->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bg0->Build();

                    // 2. Create Binding Group 1 (Read from 1, Write to 0)
                    BindingGroupHandle bgHandle1 = m_resourceManager->CreateBindingGroup( pipeHandle, 0 );
                    BindingGroup*      bg1       = m_resourceManager->GetBindingGroup( bgHandle1 );
                    bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) ); // readonly
                    bg1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) ); // writeonly
                    bg1->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bg1->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
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
                    jkrPC.fParam4              = biomechanics.dampingCoefficient;
                    jkrPC.fParam5              = biomechanics.maxRadius;
                    jkrPC.uParam0              = offsetArraySize;
                    jkrPC.uParam1              = groupIndex;
                    // domainSize.w = hash grid cell size (must match hash build cell size)
                    // Same convention as Anastomosis and Chemotaxis shaders.
                    jkrPC.domainSize           = glm::vec4( blueprint.GetDomainSize(), blueprint.GetSpatialPartitioning().cellSize );

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
                            if( g.GetCount() == 0 )
                                continue;
                            uint32_t cap = 131072;
                            while( cap < g.GetCount() )
                                cap <<= 1;
                            globalCapacity += cap;
                        }

                        size_t phenotypeSize     = globalCapacity * sizeof( uint32_t ) * 4;
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

                    std::string mitosisShaderPath = cellCycle.directedMitosis
                        ? "shaders/compute/biology/mitosis_vessel_append.comp"
                        : "shaders/compute/biology/mitosis_append.comp";
                    ComputePipelineDesc   mitosisDesc{ m_resourceManager->CreateShader( mitosisShaderPath ),
                                                     cellCycle.directedMitosis ? "MitosisVesselAppend" : "MitosisAppend" };
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

                    // Ensure vessel edge buffers exist — mitosis writes a new edge for each StalkCell division
                    if( !outState.vesselEdgeBuffer.IsValid() )
                    {
                        struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
                        outState.vesselEdgeBuffer = m_resourceManager->CreateBuffer(
                            { paddedCount * sizeof( VesselEdge ), BufferType::STORAGE, "VesselEdgeBuffer" } );
                        outState.vesselEdgeCountBuffer = m_resourceManager->CreateBuffer(
                            { sizeof( uint32_t ), BufferType::STORAGE, "VesselEdgeCountBuffer" } );
                        uint32_t zero = 0;
                        m_streamingManager->UploadBufferImmediate(
                            { { outState.vesselEdgeCountBuffer, &zero, sizeof( uint32_t ), 0 } } );
                    }

                    // Create Binding Groups (Mitosis)
                    BindingGroup* bgMitosis0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( mitosisPipeHandle, 0 ) );
                    bgMitosis0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgMitosis0->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgMitosis0->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgMitosis0->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgMitosis0->Bind( 4, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgMitosis0->Bind( 5, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgMitosis0->Build();

                    BindingGroup* bgMitosis1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( mitosisPipeHandle, 0 ) );
                    bgMitosis1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgMitosis1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgMitosis1->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgMitosis1->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgMitosis1->Bind( 4, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgMitosis1->Bind( 5, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
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
                    phenoPC.uParam0              = ( record.requiredCellType < 0 ) ? 0xFFFFFFFFu : static_cast<uint32_t>( record.requiredCellType );
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
                                if( g.GetCount() == 0 )
                                    continue;
                                uint32_t cap = 131072;
                                while( cap < g.GetCount() )
                                    cap <<= 1;
                                globalCapacity += cap;
                            }
                            size_t phenotypeSize     = globalCapacity * sizeof( uint32_t ) * 4;
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
                        pc.fParam2     = static_cast<float>( record.requiredCellType );
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
                if( std::holds_alternative<Behaviours::Perfusion>( record.behaviour ) ||
                    std::holds_alternative<Behaviours::Drain>( record.behaviour ) )
                {
                    bool        isPerfusion = std::holds_alternative<Behaviours::Perfusion>( record.behaviour );
                    std::string targetName  = isPerfusion ? std::get<Behaviours::Perfusion>( record.behaviour ).fieldName
                                                          : std::get<Behaviours::Drain>( record.behaviour ).fieldName;

                    // Positive for Perfusion (inject), negative for Drain (remove)
                    float rate = isPerfusion ? +std::get<Behaviours::Perfusion>( record.behaviour ).rate
                                            : -std::get<Behaviours::Drain>( record.behaviour ).rate;

                    GridFieldState* targetGrid = nullptr;
                    for( auto& grid: outState.gridFields )
                        if( grid.name == targetName ) { targetGrid = &grid; break; }

                    if( targetGrid )
                    {
                        // phenotypeBuffer must exist to filter by cellType == StalkCell
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

                        ShaderHandle          shaderHandle = m_resourceManager->CreateShader( "shaders/compute/biology/perfusion.comp" );
                        ComputePipelineDesc   compDesc{};
                        compDesc.shader                  = shaderHandle;
                        ComputePipelineHandle pipeHandle = m_resourceManager->CreatePipeline( compDesc );
                        ComputePipeline*      pipe       = m_resourceManager->GetPipeline( pipeHandle );

                        BindingGroup* bg0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( pipeHandle, 0 ) );
                        bg0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg0->Bind( 1, m_resourceManager->GetBuffer( targetGrid->interactionDeltaBuffer ) );
                        bg0->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        bg0->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        bg0->Build();

                        BindingGroup* bg1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( pipeHandle, 0 ) );
                        bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Bind( 1, m_resourceManager->GetBuffer( targetGrid->interactionDeltaBuffer ) );
                        bg1->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        bg1->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        bg1->Build();

                        ComputePushConstants pc{};
                        pc.fParam0     = rate;
                        pc.offset      = currentOffset;
                        pc.maxCapacity = paddedCount;
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth, 0 );

                        glm::uvec3  agentDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask perfTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch );
                        perfTask.SetTag( ( isPerfusion ? "perfusion" : "drain" ) + tagBase );
                        outState.computeGraph.AddTask( perfTask );

                        DT_INFO( "[SimulationBuilder] Compiled {} for field '{}' at {}Hz",
                                 isPerfusion ? "Perfusion" : "Drain", targetName, record.targetHz );
                    }
                    else
                    {
                        DT_WARN( "[SimulationBuilder] Target grid '{}' not found for {}!", targetName,
                                 isPerfusion ? "Perfusion" : "Drain" );
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
                        // Chemotaxis always needs the phenotype buffer (dead-cell guard + cellType filter)
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
                            size_t phenotypeSize     = globalCapacity * sizeof( uint32_t ) * 4;
                            outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );
                            struct PhenotypeData
                            {
                                uint32_t lifecycleState;
                                float    biomass;
                                float    timer;
                                uint32_t cellType;
                            };
                            std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0, 0.5f, 0.0f, 0 } );
                            m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                        }

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
                        bg0->Bind( 5, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                        bg0->Bind( 6, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
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
                        bg1->Bind( 5, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                        bg1->Bind( 6, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                        bg1->Build();

                        bool enableContactInhibition = ( chemo.contactInhibitionDensity > 0.0f );
                        float hashCellSize = blueprint.GetSpatialPartitioning().cellSize;

                        ComputePushConstants pc{};
                        pc.fParam0     = chemo.chemotacticSensitivity;
                        pc.fParam1     = chemo.receptorSaturation;
                        pc.fParam2     = chemo.maxVelocity;
                        pc.fParam3     = static_cast<float>( record.requiredCellType );
                        pc.fParam4     = chemo.contactInhibitionDensity;
                        pc.offset      = currentOffset;
                        pc.maxCapacity = paddedCount;
                        pc.uParam0     = static_cast<uint32_t>( record.requiredLifecycleState ); // -1 → 0xFFFFFFFF
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), enableContactInhibition ? hashCellSize : 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth,
                                                     enableContactInhibition ? offsetArraySize : 0u );

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
                if( std::holds_alternative<Behaviours::PhalanxActivation>( record.behaviour ) )
                {
                    const auto& phalanx = std::get<Behaviours::PhalanxActivation>( record.behaviour );

                    // Ensure phenotype buffer exists
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
                        struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
                        size_t phenotypeSize     = globalCapacity * sizeof( PhenotypeData );
                        outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );
                        std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0u, 0.5f, 0.0f, 0u } );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                    }

                    // Override cellType to PhalanxCell (3) for this group's slots
                    {
                        struct PhenotypeData { uint32_t lifecycleState; float biomass; float timer; uint32_t cellType; };
                        std::vector<PhenotypeData> phalanxInit( paddedCount, { 0u, 0.5f, 0.0f, 3u } );
                        size_t slotByteOffset = currentOffset * sizeof( PhenotypeData );
                        size_t slotByteSize   = paddedCount   * sizeof( PhenotypeData );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, phalanxInit.data(), slotByteSize, slotByteOffset } } );
                    }

                    // VEGF field — real field or 1×1×1 dummy (w=0 disables transitions in shader)
                    GridFieldState* vegfGrid        = nullptr;
                    TextureHandle   dummyVEGF;
                    glm::uvec4      phalanxGridSize{ 1u, 1u, 1u, 0u };

                    if( !phalanx.vegfFieldName.empty() )
                    {
                        for( auto& grid: outState.gridFields )
                            if( grid.name == phalanx.vegfFieldName ) { vegfGrid = &grid; break; }

                        if( vegfGrid )
                            phalanxGridSize = glm::uvec4( vegfGrid->width, vegfGrid->height, vegfGrid->depth, 1u );
                        else
                            DT_WARN( "[SimulationBuilder] PhalanxActivation: VEGF field '{}' not found — no transitions will occur", phalanx.vegfFieldName );
                    }

                    if( !vegfGrid )
                    {
                        TextureDesc dummyDesc{ 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                              TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, "PhalanxVEGF_Dummy" };
                        dummyVEGF  = m_resourceManager->CreateTexture( dummyDesc );
                        float zero = 0.0f;
                        m_streamingManager->UploadTextureImmediate( dummyVEGF, &zero, sizeof( float ) );
                    }

                    auto vegfTex0 = vegfGrid ? vegfGrid->textures[ 0 ] : dummyVEGF;
                    auto vegfTex1 = vegfGrid ? vegfGrid->textures[ 1 ] : dummyVEGF;

                    ComputePipelineDesc   phalanxDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/phalanx_activation.comp" ), "PhalanxActivationPipeline" };
                    ComputePipelineHandle phalanxPipeHandle = m_resourceManager->CreatePipeline( phalanxDesc );
                    ComputePipeline*      phalanxPipe       = m_resourceManager->GetPipeline( phalanxPipeHandle );

                    BindingGroup* bgPhalanx0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( phalanxPipeHandle, 0 ) );
                    bgPhalanx0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgPhalanx0->Bind( 1, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgPhalanx0->Bind( 2, m_resourceManager->GetTexture( vegfTex0 ), VK_IMAGE_LAYOUT_GENERAL );
                    bgPhalanx0->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgPhalanx0->Build();

                    BindingGroup* bgPhalanx1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( phalanxPipeHandle, 0 ) );
                    bgPhalanx1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgPhalanx1->Bind( 1, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgPhalanx1->Bind( 2, m_resourceManager->GetTexture( vegfTex1 ), VK_IMAGE_LAYOUT_GENERAL );
                    bgPhalanx1->Bind( 3, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgPhalanx1->Build();

                    ComputePushConstants phalanxPC{};
                    phalanxPC.fParam0     = phalanx.activationThreshold;
                    phalanxPC.fParam1     = phalanx.deactivationThreshold;
                    phalanxPC.offset      = currentOffset;
                    phalanxPC.maxCapacity = paddedCount;
                    phalanxPC.uParam0     = groupIndex;
                    phalanxPC.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                    phalanxPC.gridSize    = phalanxGridSize;

                    glm::uvec3  phalanxDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask phalanxTask( phalanxPipe, bgPhalanx0, bgPhalanx1, record.targetHz, phalanxPC, phalanxDispatch );
                    phalanxTask.SetTag( "phalanx" + tagBase );
                    outState.computeGraph.AddTask( phalanxTask );

                    DT_INFO( "[SimulationBuilder] Compiled PhalanxActivation for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::NotchDll4>( record.behaviour ) )
                {
                    const auto& notch = std::get<Behaviours::NotchDll4>( record.behaviour );

                    // Allocate vessel edge buffers if not yet created (e.g., VesselSeed comes later
                    // in the behaviour list, or is absent in tests with isolated agents).
                    if( !outState.vesselEdgeBuffer.IsValid() )
                    {
                        struct VesselEdge { uint32_t agentA, agentB; float dist; uint32_t flags; };
                        size_t edgeBufferSize     = paddedCount * sizeof( VesselEdge );
                        outState.vesselEdgeBuffer = m_resourceManager->CreateBuffer(
                            { edgeBufferSize, BufferType::STORAGE, "VesselEdgeBuffer" } );

                        outState.vesselEdgeCountBuffer = m_resourceManager->CreateBuffer(
                            { sizeof( uint32_t ), BufferType::STORAGE, "VesselEdgeCountBuffer" } );
                        uint32_t zero = 0;
                        m_streamingManager->UploadBufferImmediate(
                            { { outState.vesselEdgeCountBuffer, &zero, sizeof( uint32_t ) } } );
                    }

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

                        // Initial state: dll4=0.2 ± noise — all cells start BELOW tipThreshold (0.55)
                        // so all vessel cells begin as StalkCells. The Turing-unstable ODE then
                        // amplifies the noise asymmetry and selects exactly one TipCell per vessel.
                        std::vector<SignalingData> initSignaling( globalCapacity, { 0.2f, 0.0f, 1.0f, 0.0f } );
                        std::mt19937                          rng( 42 );
                        std::uniform_real_distribution<float> noise( -0.15f, 0.15f );
                        for( auto& s : initSignaling )
                            s.dll4 = 0.2f + noise( rng );
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

                    // VEGF texture at binding 6 — use real field or a 1×1×1 constant-1.0 dummy
                    GridFieldState* vegfGrid   = nullptr;
                    TextureHandle   dummyVEGF;
                    glm::uvec4      notchGridSize{ 1u, 1u, 1u, 0u }; // w=0 → shader skips VEGF sampling

                    if( !notch.vegfFieldName.empty() )
                    {
                        for( auto& grid: outState.gridFields )
                            if( grid.name == notch.vegfFieldName ) { vegfGrid = &grid; break; }

                        if( vegfGrid )
                        {
                            notchGridSize = glm::uvec4( vegfGrid->width, vegfGrid->height, vegfGrid->depth, 1u );
                        }
                        else
                        {
                            DT_WARN( "[SimulationBuilder] NotchDll4: VEGF field '{}' not found — running without VEGF gating", notch.vegfFieldName );
                        }
                    }

                    if( !vegfGrid )
                    {
                        // Bind a 1×1×1 dummy filled with 1.0 so the descriptor is always satisfied
                        TextureDesc dummyDesc{ 1, 1, 1, TextureType::Texture3D, VK_FORMAT_R32_SFLOAT,
                                              TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, "NotchVEGF_Dummy" };
                        dummyVEGF  = m_resourceManager->CreateTexture( dummyDesc );
                        float one  = 1.0f;
                        m_streamingManager->UploadTextureImmediate( dummyVEGF, &one, sizeof( float ) );
                    }

                    auto vegfTex0 = vegfGrid ? vegfGrid->textures[ 0 ] : dummyVEGF;
                    auto vegfTex1 = vegfGrid ? vegfGrid->textures[ 1 ] : dummyVEGF;

                    // Binding groups — ping-pong on agent read buffer; signaling/phenotype are single shared buffers
                    // Notch-Delta uses vessel edges for juxtacrine neighbor discovery (bindings 3+4).
                    BindingGroup* bgNotch0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( notchPipeHandle, 0 ) );
                    bgNotch0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgNotch0->Bind( 1, m_resourceManager->GetBuffer( outState.signalingBuffer ) );
                    bgNotch0->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgNotch0->Bind( 3, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgNotch0->Bind( 4, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgNotch0->Bind( 5, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgNotch0->Bind( 6, m_resourceManager->GetTexture( vegfTex0 ), VK_IMAGE_LAYOUT_GENERAL );
                    bgNotch0->Build();

                    BindingGroup* bgNotch1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( notchPipeHandle, 0 ) );
                    bgNotch1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgNotch1->Bind( 1, m_resourceManager->GetBuffer( outState.signalingBuffer ) );
                    bgNotch1->Bind( 2, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bgNotch1->Bind( 3, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgNotch1->Bind( 4, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgNotch1->Bind( 5, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgNotch1->Bind( 6, m_resourceManager->GetTexture( vegfTex1 ), VK_IMAGE_LAYOUT_GENERAL );
                    bgNotch1->Build();

                    // Push constants (uParam0/domainSize.w unused — edge-based signaling needs no radius)
                    ComputePushConstants notchPC{};
                    notchPC.fParam0     = notch.dll4ProductionRate;
                    notchPC.fParam1     = notch.dll4DecayRate;
                    notchPC.fParam2     = notch.notchInhibitionGain;
                    notchPC.fParam3     = notch.vegfr2BaseExpression;
                    notchPC.fParam4     = notch.tipThreshold;
                    notchPC.fParam5     = notch.stalkThreshold;
                    notchPC.offset      = currentOffset;
                    notchPC.maxCapacity = paddedCount;
                    notchPC.uParam0     = 0u;
                    notchPC.uParam1     = groupIndex;
                    notchPC.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                    notchPC.gridSize    = notchGridSize;

                    glm::uvec3 notchDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    for( uint32_t step = 0; step < notch.subSteps; ++step )
                    {
                        ComputeTask notchTask( notchPipe, bgNotch0, bgNotch1, record.targetHz, notchPC, notchDispatch );
                        notchTask.SetDtScale( 1.0f / static_cast<float>( notch.subSteps ) );
                        notchTask.SetTag( "notch_" + std::to_string( step ) + tagBase );
                        outState.computeGraph.AddTask( notchTask );
                    }

                    DT_INFO( "[SimulationBuilder] Compiled NotchDll4 for '{}' at {}Hz ({} sub-steps)", group.GetName(), record.targetHz, notch.subSteps );
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

                    // Allocate vessel component buffer before creating binding groups (needed at binding 7).
                    // Labels are read by the anastomosis shader to prevent fusing already-connected cells.
                    const bool firstAnastomosis = !outState.vesselComponentBuffer.IsValid();
                    if( firstAnastomosis )
                    {
                        // Component label buffer is global — indexed by absolute agent index
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

                        size_t componentBufferSize    = globalCapacity * sizeof( uint32_t );
                        outState.vesselComponentBuffer = m_resourceManager->CreateBuffer(
                            { componentBufferSize, BufferType::STORAGE, "VesselComponentBuffer" } );

                        // Initialize: labels[i] = i  (each agent is its own component)
                        std::vector<uint32_t> initLabels( globalCapacity );
                        std::iota( initLabels.begin(), initLabels.end(), 0u );
                        m_streamingManager->UploadBufferImmediate(
                            { { outState.vesselComponentBuffer, initLabels.data(), componentBufferSize } } );
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
                    bgAnastomosis0->Bind( 7, m_resourceManager->GetBuffer( outState.vesselComponentBuffer ) );
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
                    bgAnastomosis1->Bind( 7, m_resourceManager->GetBuffer( outState.vesselComponentBuffer ) );
                    bgAnastomosis1->Build();

                    float hashCellSize = blueprint.GetSpatialPartitioning().cellSize;

                    ComputePushConstants anastomosisPC{};
                    anastomosisPC.fParam0     = anastomosis.contactDistance;
                    anastomosisPC.fParam1     = anastomosis.allowTipToStalk ? 1.0f : 0.0f;
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

                    // Vessel Connected Components: 8-pass iterative label propagation over edges.
                    // Each pass propagates the minimum label one hop further; 8 passes handles
                    // chains up to 8 anastomosis events long.
                    if( firstAnastomosis )
                    {
                        ComputePipelineDesc   vcDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/vessel_components.comp" ),
                                                      "VesselComponentsPipeline" };
                        ComputePipelineHandle vcPipeHandle = m_resourceManager->CreatePipeline( vcDesc );
                        ComputePipeline*      vcPipe       = m_resourceManager->GetPipeline( vcPipeHandle );

                        // No ping-pong: vessel_components doesn't read/write agent positions
                        BindingGroup* bgVC =
                            m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( vcPipeHandle, 0 ) );
                        bgVC->Bind( 0, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                        bgVC->Bind( 1, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                        bgVC->Bind( 2, m_resourceManager->GetBuffer( outState.vesselComponentBuffer ) );
                        bgVC->Build();

                        ComputePushConstants vcPC{};
                        vcPC.maxCapacity = paddedCount; // max edges bounded by this group's paddedCount

                        glm::uvec3 vcDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        for( int pass = 0; pass < 8; ++pass )
                        {
                            ComputeTask vcTask( vcPipe, bgVC, bgVC, record.targetHz, vcPC, vcDispatch );
                            vcTask.SetTag( "vessel_components_" + std::to_string( pass ) + tagBase );
                            outState.computeGraph.AddTask( vcTask );
                        }

                        DT_INFO( "[SimulationBuilder] Compiled VesselComponents (8 passes) for '{}'", group.GetName() );
                    }

                    DT_INFO( "[SimulationBuilder] Compiled Anastomosis for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::VesselSeed>( record.behaviour ) )
                {
                    const auto& vesselSeed = std::get<Behaviours::VesselSeed>( record.behaviour );

                    struct VesselEdge
                    {
                        uint32_t agentA;
                        uint32_t agentB;
                        float    dist;
                        uint32_t flags;
                    };

                    // Allocate vessel edge buffers if not already allocated by Anastomosis/NotchDll4.
                    // When explicit edges are present, allocate extra capacity for runtime growth.
                    if( !outState.vesselEdgeBuffer.IsValid() )
                    {
                        size_t slotCount = vesselSeed.explicitEdges.empty()
                            ? paddedCount
                            : std::max( static_cast<size_t>( paddedCount ) * 4,
                                        vesselSeed.explicitEdges.size() * 2 );
                        outState.vesselEdgeBuffer = m_resourceManager->CreateBuffer(
                            { slotCount * sizeof( VesselEdge ), BufferType::STORAGE, "VesselEdgeBuffer" } );
                        outState.vesselEdgeCountBuffer =
                            m_resourceManager->CreateBuffer( { sizeof( uint32_t ), BufferType::STORAGE, "VesselEdgeCountBuffer" } );
                    }

                    // Build seed edges
                    std::vector<VesselEdge> seedEdges;
                    if( !vesselSeed.explicitEdges.empty() )
                    {
                        // 2D ring topology: upload the explicit edge list from VesselTreeGenerator.
                        // Compute per-edge rest length from initial cell positions so the spring
                        // shader can use the correct length for each edge independently.
                        const auto& groupPositions = group.GetPositions();
                        for( const auto& [a, b] : vesselSeed.explicitEdges )
                        {
                            float edgeDist = 0.0f;
                            if( a < groupPositions.size() && b < groupPositions.size() )
                                edgeDist = glm::length( glm::vec3( groupPositions[ a ] ) - glm::vec3( groupPositions[ b ] ) );
                            seedEdges.push_back( { currentOffset + a, currentOffset + b, edgeDist, 0u } );
                        }
                    }
                    else
                    {
                        // Legacy 1D chain fallback: consecutive pairs within each contiguous segment
                        uint32_t slotOffset = 0;
                        for( uint32_t segCount : vesselSeed.segmentCounts )
                        {
                            for( uint32_t i = 0; i + 1 < segCount; ++i )
                                seedEdges.push_back( { currentOffset + slotOffset + i, currentOffset + slotOffset + i + 1, 0.0f, 0u } );
                            slotOffset += segCount;
                        }
                    }

                    // Upload edges at byte offset 0; Anastomosis will append after these at runtime
                    if( !seedEdges.empty() )
                        m_streamingManager->UploadBufferImmediate(
                            { { outState.vesselEdgeBuffer, seedEdges.data(), seedEdges.size() * sizeof( VesselEdge ), 0 } } );

                    uint32_t edgeCount = static_cast<uint32_t>( seedEdges.size() );
                    m_streamingManager->UploadBufferImmediate( { { outState.vesselEdgeCountBuffer, &edgeCount, sizeof( uint32_t ) } } );

                    DT_INFO( "[SimulationBuilder] VesselSeed: {} edges seeded for '{}'", edgeCount, group.GetName() );
                }
                if( std::holds_alternative<Behaviours::VesselSpring>( record.behaviour ) )
                {
                    const auto& spring = std::get<Behaviours::VesselSpring>( record.behaviour );

                    // Ensure vessel edge buffers exist (VesselSeed/Anastomosis typically create these first)
                    if( !outState.vesselEdgeBuffer.IsValid() )
                    {
                        struct VesselEdge { uint32_t agentA; uint32_t agentB; float dist; uint32_t flags; };
                        outState.vesselEdgeBuffer = m_resourceManager->CreateBuffer(
                            { paddedCount * sizeof( VesselEdge ), BufferType::STORAGE, "VesselEdgeBuffer" } );
                        outState.vesselEdgeCountBuffer = m_resourceManager->CreateBuffer(
                            { sizeof( uint32_t ), BufferType::STORAGE, "VesselEdgeCountBuffer" } );
                        uint32_t zero = 0;
                        m_streamingManager->UploadBufferImmediate( { { outState.vesselEdgeCountBuffer, &zero, sizeof( uint32_t ), 0 } } );
                    }

                    ComputePipelineDesc   springDesc{ m_resourceManager->CreateShader( "shaders/compute/biology/vessel_mechanics.comp" ),
                                                      "VesselSpringPipeline" };
                    ComputePipelineHandle springPipeHandle = m_resourceManager->CreatePipeline( springDesc );
                    ComputePipeline*      springPipe       = m_resourceManager->GetPipeline( springPipeHandle );

                    // Phenotype buffer fallback: if no behaviour created it yet, bind a dummy.
                    // The shader skips the cell-type check when reqCT == -1 (any).
                    Buffer* phenoBuf0 = outState.phenotypeBuffer.IsValid()
                        ? m_resourceManager->GetBuffer( outState.phenotypeBuffer )
                        : m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] );
                    Buffer* phenoBuf1 = outState.phenotypeBuffer.IsValid()
                        ? m_resourceManager->GetBuffer( outState.phenotypeBuffer )
                        : m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] );

                    BindingGroup* bgSpring0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( springPipeHandle, 0 ) );
                    bgSpring0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgSpring0->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgSpring0->Bind( 2, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgSpring0->Bind( 3, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgSpring0->Bind( 4, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgSpring0->Bind( 5, phenoBuf0 );
                    bgSpring0->Build();

                    BindingGroup* bgSpring1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( springPipeHandle, 0 ) );
                    bgSpring1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bgSpring1->Bind( 1, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bgSpring1->Bind( 2, m_resourceManager->GetBuffer( outState.vesselEdgeBuffer ) );
                    bgSpring1->Bind( 3, m_resourceManager->GetBuffer( outState.vesselEdgeCountBuffer ) );
                    bgSpring1->Bind( 4, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bgSpring1->Bind( 5, phenoBuf1 );
                    bgSpring1->Build();

                    ComputePushConstants springPC{};
                    springPC.offset      = currentOffset;
                    springPC.maxCapacity = paddedCount;
                    springPC.fParam0     = spring.springStiffness;
                    springPC.fParam1     = spring.restingLength;
                    springPC.fParam2     = spring.dampingCoefficient;
                    springPC.fParam3     = static_cast<float>( record.requiredCellType );
                    springPC.fParam4     = spring.anchorPhalanxCells ? 1.0f : 0.0f;
                    springPC.uParam0     = groupIndex; // grpNdx

                    glm::uvec3  dispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask springTask( springPipe, bgSpring0, bgSpring1, record.targetHz, springPC, dispatch );
                    springTask.SetTag( "spring" + tagBase );
                    springTask.SetChainFlip( true );
                    outState.computeGraph.AddTask( springTask );

                    DT_INFO( "[SimulationBuilder] Compiled VesselSpring for '{}' at {}Hz", group.GetName(), record.targetHz );
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
                        pc.fParam2              = static_cast<float>( record.requiredCellType );
                        task->UpdatePushConstants( pc );
                    }
                }
                if( std::holds_alternative<Behaviours::PhalanxActivation>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "phalanx" + tagBase );
                    if( task )
                    {
                        const auto&          phalanx = std::get<Behaviours::PhalanxActivation>( record.behaviour );
                        ComputePushConstants pc      = task->GetPushConstants();
                        pc.fParam0                   = phalanx.activationThreshold;
                        pc.fParam1                   = phalanx.deactivationThreshold;
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
                    const auto& notch = std::get<Behaviours::NotchDll4>( record.behaviour );
                    for( uint32_t step = 0; step < notch.subSteps; ++step )
                    {
                        ComputeTask* task = state.computeGraph.FindTask( "notch_" + std::to_string( step ) + tagBase );
                        if( task )
                        {
                            ComputePushConstants pc = task->GetPushConstants();
                            pc.fParam0              = notch.dll4ProductionRate;
                            pc.fParam1              = notch.dll4DecayRate;
                            pc.fParam2              = notch.notchInhibitionGain;
                            pc.fParam3              = notch.vegfr2BaseExpression;
                            pc.fParam4              = notch.tipThreshold;
                            pc.fParam5              = notch.stalkThreshold;
                            task->UpdatePushConstants( pc );
                        }
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
                        pc.fParam1                       = anastomosis.allowTipToStalk ? 1.0f : 0.0f;
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::VesselSpring>( record.behaviour ) )
                {
                    ComputeTask* task = state.computeGraph.FindTask( "spring" + tagBase );
                    if( task )
                    {
                        const auto&          spring = std::get<Behaviours::VesselSpring>( record.behaviour );
                        ComputePushConstants pc     = task->GetPushConstants();
                        pc.fParam0                  = spring.springStiffness;
                        pc.fParam1                  = spring.restingLength;
                        pc.fParam2                  = spring.dampingCoefficient;
                        pc.fParam3                  = static_cast<float>( record.requiredCellType );
                        pc.fParam4                  = spring.anchorPhalanxCells ? 1.0f : 0.0f;
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::Perfusion>( record.behaviour ) ||
                         std::holds_alternative<Behaviours::Drain>( record.behaviour ) )
                {
                    bool         isPerfusion = std::holds_alternative<Behaviours::Perfusion>( record.behaviour );
                    ComputeTask* task        = state.computeGraph.FindTask( ( isPerfusion ? "perfusion" : "drain" ) + tagBase );
                    if( task )
                    {
                        float rate = isPerfusion ? +std::get<Behaviours::Perfusion>( record.behaviour ).rate
                                                 : -std::get<Behaviours::Drain>( record.behaviour ).rate;
                        ComputePushConstants pc = task->GetPushConstants();
                        pc.fParam0              = rate;
                        task->UpdatePushConstants( pc );
                    }
                }

                behaviourIndex++;
            }
            groupIndex++;
        }
    }

} // namespace DigitalTwin