#pragma once
#include "core/Core.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace DigitalTwin
{
    class JobSystem
    {
    public:
        using Job = std::function<void()>;

        struct Config
        {
            int  workerCount         = -1; // -1 = auto
            bool forceSingleThreaded = false;
        };

        JobSystem();
        ~JobSystem();

        /**
         * @brief Initializes the worker threads.
         * @param config setup for the Job System.
         */
        Result Initialize( const Config& config );
        void   Shutdown();

        /**
         * @brief Kick a generic job to be executed by any available worker.
         */
        void Kick( Job job );

        /**
         * @brief Dispatches a loop in parallel.
         * @param jobCount Number of iterations.
         * @param job Function taking the index [0, jobCount).
         */
        void Dispatch( uint32_t jobCount, std::function<void( uint32_t )> job );

        /**
         * @brief Kick a job that MUST be executed on the Main Thread (e.g., OS/Windowing).
         */
        void KickOnMainThread( Job job );

        /**
         * @brief Blocks the calling thread until all currently kicked jobs are finished.
         * Useful for frame synchronization.
         */
        void Wait();

        /**
         * @brief Checks if the calling thread is the Main Thread (thread that initialized the system).
         */
        bool IsMainThread() const;

        /**
         * @brief Processes jobs queued for the Main Thread.
         * Must be called explicitly in the main loop.
         */
        void ProcessMainThread();

        /**
         * @brief Helper to get the number of hardware threads active.
         */
        uint32_t GetWorkerCount() const { return static_cast<uint32_t>( m_workers.size() ); }

        bool IsSingleThreaded() const { return m_singleThreaded; }

    private:
        void WorkerLoop( uint32_t threadIndex );

    private:
        // Workers
        std::vector<std::thread> m_workers;
        bool                     m_running        = false;
        bool                     m_singleThreaded = false;
        std::thread::id          m_mainThreadID;

        // General Job Queue
        std::deque<Job>         m_jobQueue;
        std::mutex              m_queueMutex;
        std::condition_variable m_queueCv;

        // Main Thread Job Queue
        std::vector<Job> m_mainThreadQueue;
        std::mutex       m_mainThreadMutex;

        // Synchronization
        // Counts how many jobs are currently running or queued.
        // Wait() blocks until this drops to 0.
        std::atomic<int>        m_busyJobs{ 0 };
        std::condition_variable m_waitCv;
        std::mutex              m_waitMutex;
    };
} // namespace DigitalTwin