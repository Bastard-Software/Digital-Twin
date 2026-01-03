#include "core/Log.h"
#include "core/memory/MemorySystem.h"
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

using namespace DigitalTwin;

class MemorySystemTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool logInitialized = false;
        if( !logInitialized )
        {
            Log::Init();
            logInitialized = true;
        }

        m_memorySystem = std::make_unique<MemorySystem>();
        m_memorySystem->Initialize();
    }

    void TearDown() override
    {
        m_memorySystem->Shutdown();
        m_memorySystem.reset();
    }

    std::unique_ptr<MemorySystem> m_memorySystem;
};

// 1. Basic Initialization Test
TEST_F( MemorySystemTest, InitializationCheck )
{
    EXPECT_NE( m_memorySystem->GetSystemAllocator(), nullptr );
#ifdef DT_DEBUG
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), 0 );
#endif
}

// 2. Automatic Tracking on Allocate
TEST_F( MemorySystemTest, AutoTrackingOnAllocate )
{
    auto*  allocator = m_memorySystem->GetSystemAllocator();
    size_t size      = 256;

    // We just call Allocate. The allocator should auto-register this with MemorySystem.
    void* ptr = allocator->Allocate( size, __FILE__, __LINE__ );

    ASSERT_NE( ptr, nullptr );

#ifdef DT_DEBUG
    // Check if the system is aware of this block without manual tracking
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), 1 );
#endif

    allocator->Free( ptr );
}

// 3. Automatic Untracking on Free
TEST_F( MemorySystemTest, AutoUntrackingOnFree )
{
    auto* allocator = m_memorySystem->GetSystemAllocator();
    void* ptr       = allocator->Allocate( 128, __FILE__, __LINE__ );

#ifdef DT_DEBUG
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), 1 );
#endif

    // Free should auto-deregister the block
    allocator->Free( ptr );

#ifdef DT_DEBUG
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), 0 );
#endif
}

// 4. Multiple Allocations Scenario
TEST_F( MemorySystemTest, MultipleAllocationsWorkflow )
{
    auto*              allocator = m_memorySystem->GetSystemAllocator();
    std::vector<void*> ptrs;
    const int          count = 5;

    // Allocate multiple blocks
    for( int i = 0; i < count; ++i )
    {
        void* p = allocator->Allocate( 64, __FILE__, __LINE__ );
        ptrs.push_back( p );
    }

#ifdef DT_DEBUG
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), count );
#endif

    // Free first one
    allocator->Free( ptrs[ 0 ] );

#ifdef DT_DEBUG
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), count - 1 );
#endif

    // Free the rest
    for( size_t i = 1; i < ptrs.size(); ++i )
    {
        allocator->Free( ptrs[ i ] );
    }

#ifdef DT_DEBUG
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), 0 );
#endif
}

// 5. Thread Safety Test
TEST_F( MemorySystemTest, MultithreadedAutoTracking )
{
    auto* allocator = m_memorySystem->GetSystemAllocator();

    auto threadTask = [ & ]( int allocsCount ) {
        std::vector<void*> localPtrs;
        localPtrs.reserve( allocsCount );

        for( int i = 0; i < allocsCount; ++i )
        {
            // This triggers m_Mutex lock in TrackAllocation internally
            void* p = allocator->Allocate( 32, __FILE__, __LINE__ );
            localPtrs.push_back( p );
        }

        for( void* p: localPtrs )
        {
            // This triggers m_Mutex lock in TrackDeallocation internally
            allocator->Free( p );
        }
    };

    const int                numThreads      = 4;
    const int                allocsPerThread = 50;
    std::vector<std::thread> threads;

    for( int i = 0; i < numThreads; ++i )
    {
        threads.emplace_back( threadTask, allocsPerThread );
    }

    for( auto& t: threads )
    {
        t.join();
    }

#ifdef DT_DEBUG
    // If mutexes work correctly, the map should be empty and not corrupted
    EXPECT_EQ( m_memorySystem->GetAllocationCount(), 0 );
#endif
}
