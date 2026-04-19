#include "simulation/SimulationBuilder.h"

#include "core/Log.h"
#include <glm/gtc/packing.hpp>
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/BindingGroup.h"
#define NOMINMAX
#include "rhi/Buffer.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationValidator.h"
#include "simulation/Phenotype.h"
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
        if( cadherinProfileBuffer.IsValid() )
            resourceManager->DestroyBuffer( cadherinProfileBuffer );
        if( cadherinAffinityBuffer.IsValid() )
            resourceManager->DestroyBuffer( cadherinAffinityBuffer );
        if( polarityBuffer.IsValid() )
            resourceManager->DestroyBuffer( polarityBuffer );
        if( orientationBuffer.IsValid() )
            resourceManager->DestroyBuffer( orientationBuffer );
        if( contactHullBuffer.IsValid() )
            resourceManager->DestroyBuffer( contactHullBuffer );
        if( signalingBuffer.IsValid() )
            resourceManager->DestroyBuffer( signalingBuffer );
        if( agentReorderBuffer.IsValid() )
            resourceManager->DestroyBuffer( agentReorderBuffer );
        if( drawMetaBuffer.IsValid() )
            resourceManager->DestroyBuffer( drawMetaBuffer );
        if( visibilityBuffer.IsValid() )
            resourceManager->DestroyBuffer( visibilityBuffer );
        if( basementMembraneBuffer.IsValid() )
            resourceManager->DestroyBuffer( basementMembraneBuffer );

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
        phenotypeBuffer        = {};
        cadherinProfileBuffer  = {};
        cadherinAffinityBuffer = {};
        polarityBuffer         = {};
        orientationBuffer      = {};
        contactHullBuffer      = {};
        signalingBuffer        = {};
        agentReorderBuffer     = {};
        drawMetaBuffer         = {};
        visibilityBuffer       = {};
        basementMembraneBuffer = {};
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

            // --- Agent orientations ---
            // Groups with a contact hull use identity quaternion (0,0,0,1) — triggers quaternion
            // mode in geometry.vert and physics-driven rotation in jkr_forces.comp.
            // Groups without a contact hull use the stored normal (0,1,0,0) — shortest-arc mode.
            {
                const bool  hasHull      = !group.GetMorphology().contactHull.empty();
                const auto& orientations = group.GetOrientations();
                uint32_t    oriCount     = std::min( group.GetCount(), static_cast<uint32_t>( orientations.size() ) );
                megaOrientations.insert( megaOrientations.end(), orientations.begin(), orientations.begin() + oriCount );
                const glm::vec4 defaultOri = hasHull
                    ? glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f )   // identity quaternion → physics-driven
                    : glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f );  // +Y normal → shortest-arc
                for( uint32_t i = oriCount; i < capacity; ++i )
                    megaOrientations.push_back( defaultOri );
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

        // Visibility buffer: one uint32 per group, 1=visible 0=hidden.
        // UPLOAD so the CPU can write it every frame without a staging copy.
        {
            uint32_t numGroups = static_cast<uint32_t>( blueprint.GetGroups().size() );
            outState.visibilityBuffer = m_resourceManager->CreateBuffer(
                { numGroups * sizeof( uint32_t ), BufferType::HOST_STORAGE, "VisibilityBuffer" } );
            // Initialize all groups visible.
            std::vector<uint32_t> initVis( numGroups, 1u );
            Buffer* vb = m_resourceManager->GetBuffer( outState.visibilityBuffer );
            vb->Write( initVis.data(), initVis.size() * sizeof( uint32_t ), 0 );
        }

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

        // GPU Bitonic Sort requires the array size to be a power of two.
        // Use the global total across ALL agent groups so every agent is hashed,
        // regardless of which group they belong to or what offset they sit at.
        // outState.totalPaddedAgents is computed in AllocateAgentBuffers (before this call).
        uint32_t paddedCount     = 131072; // minimum — single-group lower bound
        while( paddedCount < outState.totalPaddedAgents )
            paddedCount <<= 1;
        uint32_t offsetArraySize = 262144; // 64×64×64 hash grid slots (sufficient for ≤262144 agents)

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
        glm::uvec3  hashDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
        ComputeTask hashTask( hashPipe, bgHash0, bgHash1, computeHz, hashPC, hashDispatch );
        hashTask.SetPhaseName( "Spatial Grid" );
        outState.computeGraph.AddTask( hashTask );

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
                ComputeTask sortTask( sortPipe, bgSort, bgSort, computeHz, sortPC, sortDispatch );
                sortTask.SetPhaseName( "Spatial Grid" );
                outState.computeGraph.AddTask( sortTask );
            }
        }

        // --- TASK C: BUILD OFFSETS ---
        ComputePushConstants offsetPC = basePC;
        offsetPC.maxCapacity          = paddedCount;
        glm::uvec3  offsetDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
        ComputeTask offsetTask( offsetPipe, bgOffset, bgOffset, computeHz, offsetPC, offsetDispatch );
        offsetTask.SetPhaseName( "Spatial Grid" );
        outState.computeGraph.AddTask( offsetTask );

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
            diffTask.SetPhaseName( "Diffusion" );
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

        // Per-group dispatch capacity (each group has at least 131072 padded slots).
        // groupPaddedCount drives dispatch sizes and per-agent maxCapacity for non-hash shaders.
        // globalHashCapacity drives maxCapacity for hash-scanning shaders (JKR, Anastomosis,
        // Chemotaxis) so their inner loop `for (i = startIdx; i < maxCapacity; i++)` covers
        // the full sorted hash array, which now spans all agent groups.
        uint32_t paddedCount        = 131072; // per-group minimum (overwritten per-group below)
        uint32_t globalHashCapacity = 131072;
        while( globalHashCapacity < outState.totalPaddedAgents )
            globalHashCapacity <<= 1;
        uint32_t offsetArraySize = 262144;

        // Pre-pass: if any group specifies non-zero cellType seeding — either the
        // group-level `SetInitialCellType` OR the per-cell `SetInitialCellTypes`
        // override (Item 2 Phase 2.1 bit-packed morphology index) — create the
        // phenotype buffer now with the correct values. All lazy-init checks
        // below will then skip this work.
        {
            bool anyNonDefault = false;
            for( const auto& g: blueprint.GetGroups() )
            {
                if( g.GetCount() == 0 ) continue;
                if( g.GetInitialCellType() != 0 )            { anyNonDefault = true; break; }
                if( !g.GetInitialCellTypes().empty() )       { anyNonDefault = true; break; }
            }

            if( anyNonDefault && !outState.phenotypeBuffer.IsValid() )
            {

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

                // Apply per-group initial cell types. The per-cell override
                // (`SetInitialCellTypes`) takes precedence when non-empty — used by
                // Item 2 vessel cells to tag each position with its bit-packed
                // (biologicalType | morphIdx << 16) value. Slots beyond the provided
                // vector and dead padding slots keep the group-level default.
                uint32_t off = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;

                    const auto& perCell = g.GetInitialCellTypes();
                    if( !perCell.empty() )
                    {
                        uint32_t copyN = std::min( static_cast<uint32_t>( perCell.size() ), g.GetCount() );
                        for( uint32_t i = 0; i < copyN; ++i )
                            initPhenotypes[ off + i ].cellType = perCell[ i ];
                        // Slots [copyN, cap) retain the { 0, 0.5, 0, 0 } default.
                    }
                    else if( g.GetInitialCellType() != 0 )
                    {
                        for( uint32_t i = 0; i < cap; ++i )
                            initPhenotypes[ off + i ].cellType = static_cast<uint32_t>( g.GetInitialCellType() );
                    }
                    off += cap;
                }

                m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
            }
        }

        // Pre-pass: allocate cadherin profile buffer and affinity UBO.
        // Both are always allocated so Stage 5's unified JKR shader always has valid bindings.
        {
            bool hasCadherin = false;
            for( const auto& g: blueprint.GetGroups() )
                for( const auto& record: g.GetBehaviours() )
                    if( std::holds_alternative<Behaviours::CadherinAdhesion>( record.behaviour ) )
                        { hasCadherin = true; break; }

            if( hasCadherin )
            {
                uint32_t globalCapacity = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;
                    globalCapacity += cap;
                }

                size_t profileSize             = globalCapacity * sizeof( glm::vec4 );
                outState.cadherinProfileBuffer = m_resourceManager->CreateBuffer(
                    { profileSize, BufferType::STORAGE, "CadherinProfileBuffer" } );

                std::vector<glm::vec4> profiles( globalCapacity, glm::vec4( 0.0f ) );
                uint32_t off = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;
                    glm::vec4 target = glm::vec4( 0.0f );
                    for( const auto& record: g.GetBehaviours() )
                        if( std::holds_alternative<Behaviours::CadherinAdhesion>( record.behaviour ) )
                        {
                            target = std::get<Behaviours::CadherinAdhesion>( record.behaviour ).targetExpression;
                            break;
                        }
                    for( uint32_t i = 0; i < cap; i++ )
                        profiles[ off + i ] = target;
                    off += cap;
                }
                m_streamingManager->UploadBufferImmediate(
                    { { outState.cadherinProfileBuffer, profiles.data(), profileSize, 0 } } );
            }
            else
            {
                // Dummy: one vec4 (16 bytes) — binding always valid, shader branch skips it
                glm::vec4 dummy( 0.0f );
                outState.cadherinProfileBuffer = m_resourceManager->CreateBuffer(
                    { sizeof( glm::vec4 ), BufferType::STORAGE, "CadherinProfileBuffer_Dummy" } );
                m_streamingManager->UploadBufferImmediate(
                    { { outState.cadherinProfileBuffer, &dummy, sizeof( glm::vec4 ), 0 } } );
            }

            // Affinity UBO — always the blueprint matrix (identity when cadherin unused)
            glm::mat4 affinity                  = blueprint.GetCadherinAffinityMatrix();
            outState.cadherinAffinityBuffer     = m_resourceManager->CreateBuffer(
                { sizeof( glm::mat4 ), BufferType::STORAGE, "CadherinAffinityBuffer" } );
            m_streamingManager->UploadBufferImmediate(
                { { outState.cadherinAffinityBuffer, &affinity, sizeof( glm::mat4 ), 0 } } );
        }

        // Pre-pass: basement-membrane plate buffer (global, single plate per sim).
        // Always allocated so both polarity_update (binding 6) and jkr_forces
        // (binding 12) always have a valid binding. Flags.x = 0 → shader skips
        // the plate block entirely; flags.x = 1 → plate active with params below.
        {
            // Step B — multi-plate buffer. Layout (matches shader PlateBuf):
            //   meta.x = plate count (0 = no plates)
            //   plates[2i+0] = (normal.xyz, height)
            //   plates[2i+1] = (contactStiffness, integrinAdhesion, anchorageDistance, polarityBias)
            // All BasementMembrane behaviours across all groups are collected
            // into the array, up to MAX_PLATES. This lifts the "one plate per
            // simulation" limit and enables channel / box / slab geometries
            // for ECTubeDemo and future 3D-ECM demos.
            constexpr uint32_t kMaxPlates = 8;
            struct GPUPlateBuffer {
                glm::uvec4 meta;                  // .x = count
                glm::vec4  plates[ 2 * kMaxPlates ];
            };
            GPUPlateBuffer buf{};
            buf.meta = glm::uvec4( 0u );

            uint32_t plateCount = 0;
            for( const auto& g : blueprint.GetGroups() )
            {
                for( const auto& r : g.GetBehaviours() )
                {
                    if( const auto* bm = std::get_if<Behaviours::BasementMembrane>( &r.behaviour ) )
                    {
                        if( plateCount >= kMaxPlates )
                        {
                            DT_WARN( "[SimulationBuilder] Exceeded MAX_PLATES=" + std::to_string( kMaxPlates ) +
                                     " — dropping extra BasementMembrane behaviours." );
                            break;
                        }
                        glm::vec3 n    = bm->planeNormal;
                        float     nLen = glm::length( n );
                        if( nLen > 0.0001f ) n /= nLen;

                        buf.plates[ 2 * plateCount + 0 ] = glm::vec4( n, bm->height );
                        buf.plates[ 2 * plateCount + 1 ] = glm::vec4( bm->contactStiffness,
                                                                      bm->integrinAdhesion,
                                                                      bm->anchorageDistance,
                                                                      bm->polarityBias );
                        ++plateCount;
                    }
                }
                if( plateCount >= kMaxPlates ) break;
            }
            buf.meta.x = plateCount;

            outState.basementMembraneBuffer = m_resourceManager->CreateBuffer(
                { sizeof( GPUPlateBuffer ), BufferType::STORAGE, "BasementMembraneBuffer" } );
            m_streamingManager->UploadBufferImmediate(
                { { outState.basementMembraneBuffer, &buf, sizeof( GPUPlateBuffer ), 0 } } );

            if( plateCount > 0 )
                DT_INFO( "[SimulationBuilder] Compiled " + std::to_string( plateCount ) +
                         " BasementMembrane plate(s)." );
        }

        // Pre-pass: allocate polarity buffer. Always allocated so JKR binding 9 is always valid.
        {
            bool hasPolarity = false;
            for( const auto& g: blueprint.GetGroups() )
                for( const auto& record: g.GetBehaviours() )
                    if( std::holds_alternative<Behaviours::CellPolarity>( record.behaviour ) )
                        { hasPolarity = true; break; }

            if( hasPolarity )
            {
                uint32_t globalCapacity = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;
                    globalCapacity += cap;
                }
                size_t polaritySize     = globalCapacity * sizeof( glm::vec4 );
                outState.polarityBuffer = m_resourceManager->CreateBuffer(
                    { polaritySize, BufferType::STORAGE, "PolarityBuffer" } );

                // Layout: per-group contiguous slabs at offsets matching agent buffer
                // (cap = next_pow2 ≥ count, with a 131072 floor). Default-zero so
                // unpolarised groups start unpolarised; overlay per-cell seeds where
                // a group populated `GetInitialPolarities()` (Item 2 Phase 2.3 —
                // VesselTreeGenerator seeds radial-outward polarity for mature vessels).
                std::vector<glm::vec4> initial( globalCapacity, glm::vec4( 0.0f ) );
                uint32_t               slabOffset = 0;
                for( const auto& g: blueprint.GetGroups() )
                {
                    if( g.GetCount() == 0 ) continue;
                    uint32_t cap = 131072;
                    while( cap < g.GetCount() ) cap <<= 1;

                    const auto& seeds = g.GetInitialPolarities();
                    if( !seeds.empty() )
                    {
                        uint32_t copyN = std::min<uint32_t>( g.GetCount(),
                                                             static_cast<uint32_t>( seeds.size() ) );
                        std::copy( seeds.begin(), seeds.begin() + copyN,
                                   initial.begin() + slabOffset );
                    }
                    slabOffset += cap;
                }

                m_streamingManager->UploadBufferImmediate(
                    { { outState.polarityBuffer, initial.data(), polaritySize, 0 } } );
            }
            else
            {
                glm::vec4 dummy( 0.0f );
                outState.polarityBuffer = m_resourceManager->CreateBuffer(
                    { sizeof( glm::vec4 ), BufferType::STORAGE, "PolarityBuffer_Dummy" } );
                m_streamingManager->UploadBufferImmediate( { { outState.polarityBuffer, &dummy, sizeof( glm::vec4 ), 0 } } );
            }
        }

        // Pre-pass: compile CellPolarity tasks BEFORE the main behaviour loop so they execute
        // in the compute graph before JKR, which reads the polarity buffer.
        {
            ShaderHandle              polarityShader = m_resourceManager->CreateShader( "shaders/compute/polarity_update.comp" );
            ComputePipelineDesc       polarityDesc{ polarityShader, "PolarityUpdatePipeline" };
            ComputePipelineHandle     polarityPipeHandle = m_resourceManager->CreatePipeline( polarityDesc );
            ComputePipeline*          polarityPipe       = m_resourceManager->GetPipeline( polarityPipeHandle );

            uint32_t preOffset  = 0;
            uint32_t preGroup   = 0;

            for( const auto& group: blueprint.GetGroups() )
            {
                if( group.GetCount() == 0 )
                {
                    preGroup++;
                    continue;
                }

                uint32_t grpPaddedCount = 131072;
                while( grpPaddedCount < group.GetCount() ) grpPaddedCount <<= 1;

                uint32_t bhvIdx = 0;
                for( const auto& record: group.GetBehaviours() )
                {
                    if( std::holds_alternative<Behaviours::CellPolarity>( record.behaviour ) )
                    {
                        const auto& pol = std::get<Behaviours::CellPolarity>( record.behaviour );

                        // Find interactionRadius from this group's Biomechanics
                        float interactionRadius = 1.5f; // default fallback
                        for( const auto& r: group.GetBehaviours() )
                            if( const auto* bio = std::get_if<Behaviours::Biomechanics>( &r.behaviour ) )
                                { interactionRadius = bio->maxRadius; break; }

                        // Ensure phenotype buffer exists (needed for dead-cell guard)
                        if( !outState.phenotypeBuffer.IsValid() )
                        {
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
                            std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0, 0.5f, 0.0f, 0 } );
                            m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                        }

                        BindingGroup* bg0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( polarityPipeHandle, 0 ) );
                        bg0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                        bg0->Bind( 1, m_resourceManager->GetBuffer( outState.polarityBuffer ) );
                        bg0->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        bg0->Bind( 3, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                        bg0->Bind( 4, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                        bg0->Bind( 5, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        bg0->Bind( 6, m_resourceManager->GetBuffer( outState.basementMembraneBuffer ) );
                        bg0->Build();

                        BindingGroup* bg1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( polarityPipeHandle, 0 ) );
                        bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                        bg1->Bind( 1, m_resourceManager->GetBuffer( outState.polarityBuffer ) );
                        bg1->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                        bg1->Bind( 3, m_resourceManager->GetBuffer( outState.hashBuffer ) );
                        bg1->Bind( 4, m_resourceManager->GetBuffer( outState.offsetBuffer ) );
                        bg1->Bind( 5, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                        bg1->Bind( 6, m_resourceManager->GetBuffer( outState.basementMembraneBuffer ) );
                        bg1->Build();

                        ComputePushConstants polPC{};
                        polPC.fParam0      = pol.regulationRate;
                        polPC.fParam1      = interactionRadius;
                        polPC.fParam2      = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredLifecycleState ) ) );
                        polPC.fParam3      = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        polPC.fParam4      = pol.propagationStrength; // Phase 4.5 — junctional coupling weight
                        polPC.offset       = preOffset;
                        polPC.maxCapacity  = globalHashCapacity;
                        polPC.uParam0      = offsetArraySize;
                        polPC.uParam1      = preGroup;
                        polPC.domainSize   = glm::vec4( blueprint.GetDomainSize(), blueprint.GetSpatialPartitioning().cellSize );

                        std::string tag = "polarity_" + std::to_string( preGroup ) + "_" + std::to_string( bhvIdx );
                        glm::uvec3  dispatch( ( grpPaddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask polTask( polarityPipe, bg0, bg1, record.targetHz, polPC, dispatch );
                        polTask.SetTag( tag );
                        polTask.SetPhaseName( "Polarity Pre-pass" );
                        // No ChainFlip — writes to polarityBuffer, not agent positions
                        outState.computeGraph.AddTask( polTask );

                        DT_INFO( "[SimulationBuilder] Compiled CellPolarity for '{}' at {}Hz", group.GetName(), record.targetHz );
                    }
                    bhvIdx++;
                }

                preOffset += grpPaddedCount;
                preGroup++;
            }
        }

        uint32_t behaviourIndex = 0;

        for( const auto& group: blueprint.GetGroups() )
        {
            if( group.GetCount() == 0 )
                continue;

            const std::string groupPhase = "Behaviours: " + group.GetName();

            behaviourIndex = 0;
            for( const auto& record: group.GetBehaviours() )
            {
                std::string tagBase = "_" + std::to_string( groupIndex ) + "_" + std::to_string( behaviourIndex );

                if( std::holds_alternative<Behaviours::CellPolarity>( record.behaviour ) )
                {
                    // Already compiled in the polarity pre-pass above
                    behaviourIndex++;
                    continue;
                }
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
                    pc.fParam1     = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                    pc.offset      = currentOffset;
                    pc.maxCapacity = paddedCount;
                    pc.uParam0     = static_cast<uint32_t>( record.requiredLifecycleState );
                    pc.uParam1     = groupIndex;

                    glm::uvec3 agentDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask brownianTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch );
                    brownianTask.SetTag( "brownian" + tagBase );
                    brownianTask.SetPhaseName( groupPhase );
                    brownianTask.SetChainFlip( true );
                    outState.computeGraph.AddTask( brownianTask );
                    DT_INFO( "SimulationBuilder: Compiled BrownianMotion for group '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::Biomechanics>( record.behaviour ) )
                {
                    auto& biomechanics = std::get<Behaviours::Biomechanics>( record.behaviour );

                    // 1. Load Shaders and Create Pipeline for Biomechanics
                    ComputePipelineDesc jkrDesc{};
                    jkrDesc.shader                      = m_resourceManager->CreateShader( "shaders/compute/jkr_forces.comp" );
                    jkrDesc.debugName                   = "JKRForcesPipeline";
                    ComputePipelineHandle jkrPipeHandle = m_resourceManager->CreatePipeline( jkrDesc );
                    ComputePipeline*      jkrPipe       = m_resourceManager->GetPipeline( jkrPipeHandle );

                    // 2a. Build contact hull buffer from this group's morphology.
                    //     Struct layout (GPU): vec4 meta + 16 × vec4 points = 272 bytes.
                    struct ContactHullGPU
                    {
                        glm::vec4 meta;          // x=count, y=edgeAlignStrength, z=hullExtentZ, w=hullExtentY
                        glm::vec4 points[ 16 ]; // model-space offsets + sub-sphere radii
                    };
                    ContactHullGPU hullGPU{};
                    {
                        const auto& hull      = group.GetMorphology().contactHull;
                        uint32_t    hullCount = std::min( static_cast<uint32_t>( hull.size() ), 16u );
                        hullGPU.meta          = glm::vec4( float( hullCount ),
                                                           group.GetMorphology().edgeAlignStrength,
                                                           group.GetMorphology().hullExtentZ,
                                                           group.GetMorphology().hullExtentY );
                        for( uint32_t h = 0; h < hullCount; ++h )
                            hullGPU.points[ h ] = hull[ h ];
                    }
                    if( !outState.contactHullBuffer.IsValid() )
                        outState.contactHullBuffer = m_resourceManager->CreateBuffer(
                            { sizeof( ContactHullGPU ), BufferType::STORAGE, "ContactHullBuffer" } );
                    m_streamingManager->UploadBufferImmediate(
                        { { outState.contactHullBuffer, &hullGPU, sizeof( ContactHullGPU ), 0 } } );

                    // 2b. Setup Binding Groups
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
                    bgJkr0->Bind( 7, m_resourceManager->GetBuffer( outState.cadherinProfileBuffer ) );
                    bgJkr0->Bind( 8, m_resourceManager->GetBuffer( outState.cadherinAffinityBuffer ) );
                    bgJkr0->Bind( 9, m_resourceManager->GetBuffer( outState.polarityBuffer ) );
                    bgJkr0->Bind( 10, m_resourceManager->GetBuffer( outState.orientationBuffer ) );
                    bgJkr0->Bind( 11, m_resourceManager->GetBuffer( outState.contactHullBuffer ) );
                    bgJkr0->Bind( 12, m_resourceManager->GetBuffer( outState.basementMembraneBuffer ) );
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
                    bgJkr1->Bind( 7, m_resourceManager->GetBuffer( outState.cadherinProfileBuffer ) );
                    bgJkr1->Bind( 8, m_resourceManager->GetBuffer( outState.cadherinAffinityBuffer ) );
                    bgJkr1->Bind( 9, m_resourceManager->GetBuffer( outState.polarityBuffer ) );
                    bgJkr1->Bind( 10, m_resourceManager->GetBuffer( outState.orientationBuffer ) );
                    bgJkr1->Bind( 11, m_resourceManager->GetBuffer( outState.contactHullBuffer ) );
                    bgJkr1->Bind( 12, m_resourceManager->GetBuffer( outState.basementMembraneBuffer ) );
                    bgJkr1->Build();

                    // 3. Configure Task specific parameters
                    ComputePushConstants jkrPC = basePC;
                    jkrPC.offset               = currentOffset;
                    // globalHashCapacity: the inner hash-scan loop needs to be able to reach any
                    // entry in the sorted hash array, which now spans all agent groups.
                    jkrPC.maxCapacity          = globalHashCapacity;
                    jkrPC.fParam0              = biomechanics.repulsionStiffness;
                    jkrPC.fParam1              = biomechanics.adhesionStrength;
                    jkrPC.fParam2              = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredLifecycleState ) ) );
                    jkrPC.fParam3              = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                    jkrPC.fParam4              = biomechanics.dampingCoefficient;
                    jkrPC.fParam5              = biomechanics.maxRadius;
                    jkrPC.uParam0              = offsetArraySize;
                    jkrPC.uParam1              = groupIndex;
                    // domainSize.w = hash grid cell size (must match hash build cell size)
                    // Same convention as Anastomosis and Chemotaxis shaders.
                    jkrPC.domainSize           = glm::vec4( blueprint.GetDomainSize(), blueprint.GetSpatialPartitioning().cellSize );

                    // Cadherin-scaled adhesion: check if this group also has CadherinAdhesion.
                    // gridSize.x is a composite field:
                    //   bit 0       = cadherin active flag
                    //   bits 1..8   = catchBondStrength × 51 (Phase 5; 8-bit fixed-point
                    //                 mapping [0, 5] → [0, 255])
                    //   bits 16..31 = packHalf2x16 upper half = corticalTension (Biomechanics)
                    // gridSize.y:
                    //   packHalf2x16(catchBondPeakLoad, couplingStrength) — both half-floats.
                    //   couplingStrength precision drops from fp32 → fp16 (~3 decimal digits),
                    //   acceptable for a multiplier typically in [0, 10].
                    {
                        const Behaviours::CadherinAdhesion* cadherin = nullptr;
                        for( const auto& r: group.GetBehaviours() )
                            if( const auto* c = std::get_if<Behaviours::CadherinAdhesion>( &r.behaviour ) )
                                { cadherin = c; break; }
                        uint32_t cadFlag       = cadherin ? 1u : 0u;
                        uint32_t tensionPacked = glm::packHalf2x16( glm::vec2( 0.0f, biomechanics.corticalTension ) );
                        uint32_t catchStrengthBits = 0u;
                        if( cadherin )
                        {
                            float   clamped = glm::clamp( cadherin->catchBondStrength / 5.0f, 0.0f, 1.0f );
                            catchStrengthBits = static_cast<uint32_t>( clamped * 255.0f + 0.5f ) & 0xFFu;
                        }
                        jkrPC.gridSize.x = cadFlag | ( catchStrengthBits << 1 ) | ( tensionPacked & 0xFFFF0000u );
                        jkrPC.gridSize.y = cadherin
                            ? glm::packHalf2x16( glm::vec2( cadherin->catchBondPeakLoad, cadherin->couplingStrength ) )
                            : 0u;
                    }

                    // Polarity-modulated adhesion: check if this group also has CellPolarity.
                    // gridSize.z layout (Phase 4.5-B):
                    //   bit 0       = polarity active flag
                    //   bits 16..31 = packHalf2x16 upper half = lateralAdhesionScale
                    //                 (Biomechanics cadherin-belt translational pull)
                    {
                        const Behaviours::CellPolarity* polarity = nullptr;
                        for( const auto& r: group.GetBehaviours() )
                            if( const auto* p = std::get_if<Behaviours::CellPolarity>( &r.behaviour ) )
                                { polarity = p; break; }
                        uint32_t polFlag       = polarity ? 1u : 0u;
                        uint32_t lateralPacked = glm::packHalf2x16( glm::vec2( 0.0f, biomechanics.lateralAdhesionScale ) );
                        jkrPC.gridSize.z = polFlag | ( lateralPacked & 0xFFFF0000u );
                        jkrPC.gridSize.w = polarity
                            ? glm::packHalf2x16( glm::vec2( polarity->apicalRepulsion, polarity->basalAdhesion ) )
                            : 0u;
                    }

                    // 4. Append Task to Compute Graph
                    glm::uvec3 jkrDispatch( ( paddedCount + 255 ) / 256, 1, 1 );

                    ComputeTask jkrTask( jkrPipe, bgJkr0, bgJkr1, record.targetHz, jkrPC, jkrDispatch );
                    jkrTask.SetTag( "jkr" + tagBase );
                    jkrTask.SetPhaseName( groupPhase );
                    jkrTask.SetChainFlip( true );
                    outState.computeGraph.AddTask( jkrTask );

                    DT_INFO( "[SimulationBuilder] Compiled Biomechanics (JKR) for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                if( std::holds_alternative<Behaviours::CadherinAdhesion>( record.behaviour ) )
                {
                    const auto& cadherin = std::get<Behaviours::CadherinAdhesion>( record.behaviour );

                    // Ensure phenotype buffer exists (needed for lifecycle-state filter)
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
                        std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0, 0.5f, 0.0f, 0 } );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                    }

                    ShaderHandle shaderHandle = m_resourceManager->CreateShader(
                        "shaders/compute/cadherin_expression_update.comp" );
                    ComputePipelineDesc   pipeDesc{ shaderHandle, "CadherinExpressionUpdate" };
                    ComputePipelineHandle pipeHandle = m_resourceManager->CreatePipeline( pipeDesc );
                    ComputePipeline*      pipe       = m_resourceManager->GetPipeline( pipeHandle );

                    // Two binding groups tracking the active agent buffer index (dead-cell guard only).
                    // cadherinProfileBuffer is in-place readwrite — same handle in both groups.
                    BindingGroup* bg0 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( pipeHandle, 0 ) );
                    bg0->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 0 ] ) );
                    bg0->Bind( 1, m_resourceManager->GetBuffer( outState.cadherinProfileBuffer ) );
                    bg0->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bg0->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bg0->Build();

                    BindingGroup* bg1 = m_resourceManager->GetBindingGroup( m_resourceManager->CreateBindingGroup( pipeHandle, 0 ) );
                    bg1->Bind( 0, m_resourceManager->GetBuffer( outState.agentBuffers[ 1 ] ) );
                    bg1->Bind( 1, m_resourceManager->GetBuffer( outState.cadherinProfileBuffer ) );
                    bg1->Bind( 2, m_resourceManager->GetBuffer( outState.agentCountBuffer ) );
                    bg1->Bind( 3, m_resourceManager->GetBuffer( outState.phenotypeBuffer ) );
                    bg1->Build();

                    ComputePushConstants cadPC{};
                    cadPC.fParam0     = cadherin.expressionRate;
                    cadPC.fParam1     = cadherin.degradationRate;
                    cadPC.fParam2     = cadherin.targetExpression.x;
                    cadPC.fParam3     = cadherin.targetExpression.y;
                    cadPC.fParam4     = cadherin.targetExpression.z;
                    cadPC.fParam5     = cadherin.targetExpression.w;
                    cadPC.offset      = currentOffset;
                    cadPC.maxCapacity = paddedCount;
                    cadPC.uParam0     = static_cast<uint32_t>( record.requiredLifecycleState );
                    cadPC.uParam1     = groupIndex;

                    glm::uvec3 dispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                    ComputeTask cadTask( pipe, bg0, bg1, record.targetHz, cadPC, dispatch );
                    cadTask.SetTag( "cadherin_expr" + tagBase );
                    cadTask.SetPhaseName( groupPhase );
                    // No ChainFlip: this shader writes to cadherinProfileBuffer, not agent positions.
                    outState.computeGraph.AddTask( cadTask );

                    DT_INFO( "[SimulationBuilder] Compiled CadherinAdhesion (expression) for '{}' at {}Hz",
                             group.GetName(), record.targetHz );
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

                    // Mitosis — generic append-only (Item 2 demolition 2026-04-19): the
                    // previous directedMitosis + mitosis_vessel_append.comp path depended on
                    // the pre-Item-1 vessel edge graph, now removed. Item 3's sprouting
                    // redesign will reintroduce directed mitosis on top of the cell-based
                    // substrate without a persistent edge buffer.
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

                    // Create Binding Groups (Mitosis) — generic append-only, no vessel graph.
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
                    phenoPC.uParam0              = static_cast<uint32_t>( record.requiredCellType );
                    phenoPC.uParam1              = groupIndex;

                    ComputePushConstants mitosisPC = basePC;
                    mitosisPC.offset               = currentOffset;
                    mitosisPC.maxCapacity          = paddedCount;
                    mitosisPC.uParam1              = groupIndex;

                    // 5. Append Tasks to Compute Graph
                    glm::uvec3 maxDispatch( ( paddedCount + 255 ) / 256, 1, 1 );

                    ComputeTask phenoTask( phenoPipe, bgPheno0, bgPheno1, record.targetHz, phenoPC, maxDispatch );
                    phenoTask.SetTag( "phenotype" + tagBase );
                    phenoTask.SetPhaseName( groupPhase );
                    outState.computeGraph.AddTask( phenoTask );

                    ComputeTask mitosisTask( mitosisPipe, bgMitosis0, bgMitosis1, record.targetHz, mitosisPC, maxDispatch );
                    mitosisTask.SetTag( "mitosis" + tagBase );
                    mitosisTask.SetPhaseName( groupPhase );
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
                    LifecycleState requiredLifecycleState = isConsume
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
                        if( requiredLifecycleState != LifecycleState::Any && !outState.phenotypeBuffer.IsValid() )
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
                        pc.fParam1     = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( requiredLifecycleState ) ) );
                        pc.fParam2     = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        pc.offset      = currentOffset;
                        pc.maxCapacity = paddedCount;
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth, 0 );

                        glm::uvec3 agentDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask chemTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch );
                        chemTask.SetTag( "chemfield" + tagBase );
                        chemTask.SetPhaseName( groupPhase );
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
                        pc.fParam1     = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        pc.offset      = currentOffset;
                        pc.maxCapacity = paddedCount;
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth, 0 );

                        glm::uvec3  agentDispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask perfTask( pipe, bg0, bg1, record.targetHz, pc, agentDispatch );
                        perfTask.SetTag( ( isPerfusion ? "perfusion" : "drain" ) + tagBase );
                        perfTask.SetPhaseName( groupPhase );
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
                        pc.fParam3     = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        pc.fParam4     = chemo.contactInhibitionDensity;
                        pc.offset      = currentOffset;
                        pc.maxCapacity = globalHashCapacity; // hash scan bound spans all groups
                        pc.uParam0     = static_cast<uint32_t>( record.requiredLifecycleState );
                        pc.uParam1     = groupIndex;
                        pc.domainSize  = glm::vec4( blueprint.GetDomainSize(), enableContactInhibition ? hashCellSize : 0.0f );
                        pc.gridSize    = glm::uvec4( targetGrid->width, targetGrid->height, targetGrid->depth,
                                                     enableContactInhibition ? offsetArraySize : 0u );

                        glm::uvec3  dispatch( ( paddedCount + 255 ) / 256, 1, 1 );
                        ComputeTask task( pipePtr, bg0, bg1, record.targetHz, pc, dispatch );
                        task.SetTag( "chemotaxis" + tagBase );
                        task.SetPhaseName( groupPhase );
                        task.SetChainFlip( true );
                        outState.computeGraph.AddTask( task );
                        DT_INFO( "[SimulationBuilder] Compiled Chemotaxis for '{}' -> '{}' at {}Hz",
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
                        size_t phenotypeSize     = globalCapacity * sizeof( PhenotypeData );
                        outState.phenotypeBuffer = m_resourceManager->CreateBuffer( { phenotypeSize, BufferType::STORAGE, "PhenotypeBuffer" } );
                        std::vector<PhenotypeData> initPhenotypes( globalCapacity, { 0u, 0.5f, 0.0f, 0u } );
                        m_streamingManager->UploadBufferImmediate( { { outState.phenotypeBuffer, initPhenotypes.data(), phenotypeSize, 0 } } );
                    }

                    // Override cellType to PhalanxCell (3) for this group's slots
                    {
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
                                              TextureUsage::STORAGE | TextureUsage::TRANSFER_DST, VK_SAMPLE_COUNT_1_BIT, "PhalanxVEGF_Dummy" };
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
                    phalanxTask.SetPhaseName( groupPhase );
                    outState.computeGraph.AddTask( phalanxTask );

                    DT_INFO( "[SimulationBuilder] Compiled PhalanxActivation for '{}' at {}Hz", group.GetName(), record.targetHz );
                }
                // NotchDll4 / Anastomosis / VesselSeed / VesselSpring branches removed
                // in Item 2 demolition (2026-04-19). The pre-Item-1 vessel-graph
                // infrastructure (edge buffers, spring forces, component labelling,
                // juxtacrine-via-edges signalling) has been demolished; cell-based
                // physics from Item 1 (JKR + VE-cadherin catch-bond + lateral adhesion
                // + BM-gated polarity; Rakshit 2012; Halbleib & Nelson 2006; Strilic 2009)
                // now holds vessels together without any persistent graph. Item 3's
                // sprouting redesign will reintroduce tip/stalk differentiation on the
                // cell-based substrate without edge buffers.
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
                        pc.fParam1                 = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
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
                        pc.fParam2                = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredLifecycleState ) ) );
                        pc.fParam3                = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        pc.fParam4                = bio.dampingCoefficient;
                        pc.fParam5                = bio.maxRadius;
                        // domainSize.w holds the spatial hash cell size — do NOT overwrite with maxRadius
                        // Update cadherin coupling flag + Phase 5 catch-bond params in case
                        // anything changed via HotReload. gridSize.x / .y layout mirrors the
                        // initial compile path:
                        //   x: bit 0 = cadherin flag; bits 1..8 = catchBondStrength × 51;
                        //      bits 16..31 = corticalTension (half float).
                        //   y: packHalf2x16(catchBondPeakLoad, couplingStrength)
                        {
                            const Behaviours::CadherinAdhesion* cadherin = nullptr;
                            for( const auto& r: group.GetBehaviours() )
                                if( const auto* c = std::get_if<Behaviours::CadherinAdhesion>( &r.behaviour ) )
                                    { cadherin = c; break; }
                            uint32_t cadFlag       = cadherin ? 1u : 0u;
                            uint32_t tensionPacked = glm::packHalf2x16( glm::vec2( 0.0f, bio.corticalTension ) );
                            uint32_t catchStrengthBits = 0u;
                            if( cadherin )
                            {
                                float   clamped = glm::clamp( cadherin->catchBondStrength / 5.0f, 0.0f, 1.0f );
                                catchStrengthBits = static_cast<uint32_t>( clamped * 255.0f + 0.5f ) & 0xFFu;
                            }
                            pc.gridSize.x = cadFlag | ( catchStrengthBits << 1 ) | ( tensionPacked & 0xFFFF0000u );
                            pc.gridSize.y = cadherin
                                ? glm::packHalf2x16( glm::vec2( cadherin->catchBondPeakLoad, cadherin->couplingStrength ) )
                                : 0u;
                        }
                        // Update polarity flag + lateralAdhesionScale (Phase 4.5-B) in case
                        // any parameter changed via HotReload. gridSize.z: bit 0 = polarity
                        // active flag; bits 16..31 = half-packed lateralAdhesionScale.
                        {
                            const Behaviours::CellPolarity* polarity = nullptr;
                            for( const auto& r: group.GetBehaviours() )
                                if( const auto* p = std::get_if<Behaviours::CellPolarity>( &r.behaviour ) )
                                    { polarity = p; break; }
                            uint32_t polFlag       = polarity ? 1u : 0u;
                            uint32_t lateralPacked = glm::packHalf2x16( glm::vec2( 0.0f, bio.lateralAdhesionScale ) );
                            pc.gridSize.z = polFlag | ( lateralPacked & 0xFFFF0000u );
                            pc.gridSize.w = polarity
                                ? glm::packHalf2x16( glm::vec2( polarity->apicalRepulsion, polarity->basalAdhesion ) )
                                : 0u;
                        }
                        task->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::CadherinAdhesion>( record.behaviour ) )
                {
                    const auto& cad = std::get<Behaviours::CadherinAdhesion>( record.behaviour );

                    ComputeTask* task = state.computeGraph.FindTask( "cadherin_expr" + tagBase );
                    if( task )
                    {
                        ComputePushConstants pc   = task->GetPushConstants();
                        pc.fParam0               = cad.expressionRate;
                        pc.fParam1               = cad.degradationRate;
                        pc.fParam2               = cad.targetExpression.x;
                        pc.fParam3               = cad.targetExpression.y;
                        pc.fParam4               = cad.targetExpression.z;
                        pc.fParam5               = cad.targetExpression.w;
                        pc.uParam0               = static_cast<uint32_t>( record.requiredLifecycleState );
                        task->UpdatePushConstants( pc );
                    }

                    // Phase 5 — propagate couplingStrength / catchBondStrength /
                    // catchBondPeakLoad into the JKR task's push constants when the
                    // user edits CadherinAdhesion via the inspector. Mirrors the
                    // encoding in the initial-compile + Biomechanics hot-reload paths.
                    ComputeTask* jkrTask = state.computeGraph.FindTask( "jkr" + tagBase );
                    if( jkrTask )
                    {
                        ComputePushConstants pc = jkrTask->GetPushConstants();
                        float corticalTension = 0.0f;
                        float lateralScale    = 0.0f;
                        for( const auto& r: group.GetBehaviours() )
                            if( const auto* bio = std::get_if<Behaviours::Biomechanics>( &r.behaviour ) )
                            { corticalTension = bio->corticalTension; lateralScale = bio->lateralAdhesionScale; break; }
                        uint32_t tensionPacked = glm::packHalf2x16( glm::vec2( 0.0f, corticalTension ) );
                        float    clamped = glm::clamp( cad.catchBondStrength / 5.0f, 0.0f, 1.0f );
                        uint32_t catchStrengthBits = static_cast<uint32_t>( clamped * 255.0f + 0.5f ) & 0xFFu;
                        pc.gridSize.x = 1u | ( catchStrengthBits << 1 ) | ( tensionPacked & 0xFFFF0000u );
                        pc.gridSize.y = glm::packHalf2x16( glm::vec2( cad.catchBondPeakLoad, cad.couplingStrength ) );
                        jkrTask->UpdatePushConstants( pc );
                    }
                }
                else if( std::holds_alternative<Behaviours::CellPolarity>( record.behaviour ) )
                {
                    // Tag uses groupIndex/behaviourIndex matching the pre-pass loop
                    std::string polarTag = "polarity_" + std::to_string( groupIndex ) + "_" + std::to_string( behaviourIndex );
                    ComputeTask* task    = state.computeGraph.FindTask( polarTag );
                    if( task )
                    {
                        const auto&          pol = std::get<Behaviours::CellPolarity>( record.behaviour );
                        ComputePushConstants pc  = task->GetPushConstants();
                        pc.fParam0               = pol.regulationRate;
                        // fParam1 (interactionRadius) comes from the group's Biomechanics
                        for( const auto& r: group.GetBehaviours() )
                            if( const auto* bio = std::get_if<Behaviours::Biomechanics>( &r.behaviour ) )
                                { pc.fParam1 = bio->maxRadius; break; }
                        pc.fParam2               = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredLifecycleState ) ) );
                        pc.fParam3               = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        pc.fParam4               = pol.propagationStrength; // Phase 4.5 — hot-reload junctional coupling weight
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
                        LifecycleState reqLifecycleState = isConsume
                                                      ? std::get<Behaviours::ConsumeField>( record.behaviour ).requiredLifecycleState
                                                      : std::get<Behaviours::SecreteField>( record.behaviour ).requiredLifecycleState;

                        ComputePushConstants pc = task->GetPushConstants();
                        pc.fParam0              = rate;
                        pc.fParam1              = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( reqLifecycleState ) ) );
                        pc.fParam2              = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
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
                        pc.fParam3                 = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        pc.uParam0                 = static_cast<uint32_t>( record.requiredLifecycleState );
                        task->UpdatePushConstants( pc );
                    }
                }
                // NotchDll4 / Anastomosis / VesselSpring hot-reload branches removed
                // in Item 2 demolition — their behaviours no longer exist.
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
                        pc.fParam1              = static_cast<float>( static_cast<int32_t>( static_cast<uint32_t>( record.requiredCellType ) ) );
                        task->UpdatePushConstants( pc );
                    }
                }

                behaviourIndex++;
            }
            groupIndex++;
        }

        // ── Basement-membrane plates (hot reload, Step B) ─────────────────────
        // Scan for ALL BasementMembrane behaviours across all groups. Re-upload
        // the plate buffer with current parameters. Enables multi-plate live-
        // editing and toggling plates off by removing behaviours between reloads.
        if( state.basementMembraneBuffer.IsValid() )
        {
            constexpr uint32_t kMaxPlates = 8;
            struct GPUPlateBuffer {
                glm::uvec4 meta;
                glm::vec4  plates[ 2 * kMaxPlates ];
            };
            GPUPlateBuffer buf{};
            buf.meta = glm::uvec4( 0u );

            uint32_t plateCount = 0;
            for( const auto& g : blueprint.GetGroups() )
            {
                for( const auto& r : g.GetBehaviours() )
                {
                    if( const auto* bm = std::get_if<Behaviours::BasementMembrane>( &r.behaviour ) )
                    {
                        if( plateCount >= kMaxPlates ) break;
                        glm::vec3 n    = bm->planeNormal;
                        float     nLen = glm::length( n );
                        if( nLen > 0.0001f ) n /= nLen;

                        buf.plates[ 2 * plateCount + 0 ] = glm::vec4( n, bm->height );
                        buf.plates[ 2 * plateCount + 1 ] = glm::vec4( bm->contactStiffness,
                                                                      bm->integrinAdhesion,
                                                                      bm->anchorageDistance,
                                                                      bm->polarityBias );
                        ++plateCount;
                    }
                }
                if( plateCount >= kMaxPlates ) break;
            }
            buf.meta.x = plateCount;

            m_streamingManager->UploadBufferImmediate(
                { { state.basementMembraneBuffer, &buf, sizeof( GPUPlateBuffer ), 0 } } );
        }
    }

} // namespace DigitalTwin