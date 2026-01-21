#include "core/jobs/JobSystem.h"
#include <atomic>
#include <gtest/gtest.h>
#include <vector>

using namespace DigitalTwin;

class JobSystemTest : public ::testing::Test
{
protected:
    std::unique_ptr<JobSystem> jobs;

    void SetUp() override { jobs = std::make_unique<JobSystem>(); }

    void TearDown() override { jobs->Shutdown(); }
};

// 1. Test Single Job Execution (Multi-threaded)
TEST_F( JobSystemTest, KickSingleJob )
{
    JobSystem::Config config;
    jobs->Initialize( config );

    std::atomic<bool> done = false;

    jobs->Kick( [ &done ]() {
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        done = true;
    } );

    jobs->Wait();
    EXPECT_TRUE( done );
}

// 2. Test Parallel Dispatch
TEST_F( JobSystemTest, DispatchParallel )
{
    JobSystem::Config config;
    jobs->Initialize( config );

    const int        count   = 100;
    std::atomic<int> counter = 0;
    std::vector<int> results( count, 0 );

    jobs->Dispatch( count, [ & ]( uint32_t index ) {
        counter.fetch_add( 1 );
        results[ index ] = index * 2;
    } );

    jobs->Wait();

    EXPECT_EQ( counter, count );
    for( int i = 0; i < count; ++i )
    {
        EXPECT_EQ( results[ i ], i * 2 );
    }
}

// 3. Test Main Thread Affinity
TEST_F( JobSystemTest, MainThreadExecution )
{
    JobSystem::Config config;
    jobs->Initialize( config );

    std::thread::id   mainThreadId = std::this_thread::get_id();
    std::atomic<bool> executed     = false;
    std::thread::id   jobThreadId;

    jobs->KickOnMainThread( [ & ]() {
        jobThreadId = std::this_thread::get_id();
        executed    = true;
    } );

    EXPECT_FALSE( executed );
    jobs->ProcessMainThread();
    EXPECT_TRUE( executed );
    EXPECT_EQ( jobThreadId, mainThreadId ) << "Job executed on wrong thread ID";
}

// 4. Test FORCE SINGLE THREADED Mode
TEST_F( JobSystemTest, ForceSingleThreaded )
{
    JobSystem::Config config;
    config.forceSingleThreaded = true; // <--- MODE ENABLED

    jobs->Initialize( config );

    EXPECT_TRUE( jobs->IsSingleThreaded() );
    EXPECT_EQ( jobs->GetWorkerCount(), 0 );

    std::thread::id mainThreadId = std::this_thread::get_id();
    std::thread::id jobThreadId;
    bool            executed = false;

    // Kick should execute immediately
    jobs->Kick( [ & ]() {
        jobThreadId = std::this_thread::get_id();
        executed    = true;
    } );

    EXPECT_TRUE( executed ) << "Job should have executed immediately";
    EXPECT_EQ( jobThreadId, mainThreadId ) << "Job should execute on main thread";
}