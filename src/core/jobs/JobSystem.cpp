#include "core/jobs/JobSystem.h"

#include "core/Log.h"

namespace DigitalTwin
{

    JobSystem::JobSystem()
    {
    }

    JobSystem::~JobSystem()
    {
    }

    Result JobSystem::Initialize( const Config& config )
    {
        if( m_running )
        {
            DT_WARN( "JobSystem is already initialized." );
            return Result::SUCCESS;
        }

        m_running        = true;
        m_singleThreaded = config.forceSingleThreaded;
        m_mainThreadID   = std::this_thread::get_id();

        if( m_singleThreaded )
        {
            DT_WARN( "JobSystem initialized in FORCE SINGLE THREADED mode." );
            // We don't spawn any workers.
            // Kick() will execute jobs immediately on the calling thread.
            return Result::SUCCESS;
        }

        // Calculate worker count
        uint32_t workerCount;
        if( config.workerCount > 0 )
        {
            workerCount = static_cast<uint32_t>( config.workerCount );
        }
        else
        {
            // Auto-detect: Leave one core free for Main Thread if possible
            unsigned int hardware = std::thread::hardware_concurrency();
            workerCount           = ( hardware > 1 ) ? hardware - 1 : 1;
        }

        DT_INFO( "Initializing JobSystem with {} worker threads.", workerCount );

        m_workers.reserve( workerCount );
        for( uint32_t i = 0; i < workerCount; ++i )
        {
            // Thread index 1..N (0 is reserved for Main Thread)
            uint32_t threadIndex = i + 1;
            m_workers.emplace_back( &JobSystem::WorkerLoop, this, threadIndex );
        }

        return Result::SUCCESS;
    }

    void JobSystem::Shutdown()
    {
        if( !m_running )
            return;

        {
            std::lock_guard<std::mutex> lock( m_queueMutex );
            m_running = false;
        }

        // Wake up all threads so they can finish and exit
        m_queueCv.notify_all();

        for( std::thread& worker: m_workers )
        {
            if( worker.joinable() )
                worker.join();
        }

        m_workers.clear();
        m_jobQueue.clear();
        m_mainThreadQueue.clear();
        m_busyJobs = 0;
    }

    void JobSystem::Kick( Job job )
    {
        if( m_singleThreaded )
        {
            // Execute immediately on the calling thread (usually Main)
            job();
            return;
        }

        // Increment busy counter BEFORE pushing to queue
        m_busyJobs.fetch_add( 1 );

        {
            std::lock_guard<std::mutex> lock( m_queueMutex );
            m_jobQueue.push_back( std::move( job ) );
        }
        m_queueCv.notify_one();
    }

    void JobSystem::Dispatch( uint32_t jobCount, std::function<void( uint32_t )> job )
    {
        if( jobCount == 0 )
            return;

        if( m_singleThreaded )
        {
            // Execute loop serially
            for( uint32_t i = 0; i < jobCount; ++i )
            {
                job( i );
            }
            return;
        }

        // Increment busy counter by total jobs
        m_busyJobs.fetch_add( jobCount );

        {
            std::lock_guard<std::mutex> lock( m_queueMutex );
            for( uint32_t i = 0; i < jobCount; ++i )
            {
                // Capture i by value
                m_jobQueue.push_back( [ job, i ]() { job( i ); } );
            }
        }
        m_queueCv.notify_all();
    }

    void JobSystem::KickOnMainThread( Job job )
    {
        // Note: We don't increment m_busyJobs here strictly for Wait() logic usually,
        // but if we want Wait() to wait for MainThread jobs too, we should.
        // For now, let's assume Wait() is primarily for worker tasks.

        std::lock_guard<std::mutex> lock( m_mainThreadMutex );
        m_mainThreadQueue.push_back( std::move( job ) );
    }

    void JobSystem::Wait()
    {
        if( m_singleThreaded )
            return; // Nothing to wait for, jobs are already done

        // If we are on Main Thread, we should help work while waiting!
        // TODO: Implement Work Stealing here.

        std::unique_lock<std::mutex> lock( m_waitMutex );
        m_waitCv.wait( lock, [ this ]() { return m_busyJobs.load() == 0; } );
    }

    bool JobSystem::IsMainThread() const
    {
        return std::this_thread::get_id() == m_mainThreadID;
    }

    void JobSystem::ProcessMainThread()
    {
        // Swap queue to avoid locking during execution
        std::vector<Job> currentJobs;
        {
            std::lock_guard<std::mutex> lock( m_mainThreadMutex );
            if( m_mainThreadQueue.empty() )
                return;
            currentJobs.swap( m_mainThreadQueue );
        }

        for( const auto& job: currentJobs )
        {
            job();
        }
    }

    void JobSystem::WorkerLoop( uint32_t threadIndex )
    {
        // Initialization of thread-local resources (e.g. RHI Context)
        // will happen lazily upon first access via RHI::GetThreadContext().

        // 2. Work Loop
        while( true )
        {
            Job job;
            {
                std::unique_lock<std::mutex> lock( m_queueMutex );

                // Wait until there is a job or we are stopping
                m_queueCv.wait( lock, [ this ]() { return !m_jobQueue.empty() || !m_running; } );

                if( !m_running && m_jobQueue.empty() )
                    break;

                job = std::move( m_jobQueue.front() );
                m_jobQueue.pop_front();
            }

            // Execute the job outside the lock
            job();

            // Decrement busy counter
            int remaining = m_busyJobs.fetch_sub( 1 ) - 1;
            if( remaining == 0 )
            {
                std::lock_guard<std::mutex> lock( m_waitMutex );
                m_waitCv.notify_all();
            }
        }
    }

} // namespace DigitalTwin