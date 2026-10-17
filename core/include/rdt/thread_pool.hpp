#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// A small std::thread-based pool -- no external dependency, matching the
// header-only philosophy of the rest of the core. Used by Random Forest to
// build trees in parallel, and by Gradient Boosting to build a multiclass
// round's per-class trees in parallel (rounds themselves stay sequential).
namespace rdt {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t n_threads) {
        for (std::size_t i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    std::size_t size() const { return workers_.size(); }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using Ret = decltype(f());
        auto task = std::make_shared<std::packaged_task<Ret()>>(std::forward<F>(f));
        std::future<Ret> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // Run `count` independent jobs (indices 0..count-1) across the pool and
    // block until all finish. Simpler than submit()-ing N futures by hand
    // for the common "parallel for" case that Random Forest/multiclass
    // boosting rounds need.
    //
    // Waits for every job even if one throws (the jobs reference the
    // caller's locals), then rethrows the first exception.
    template <typename F>
    void parallel_for(std::size_t count, F&& f) {
        if (count == 0) return;
        std::vector<std::future<void>> futures;
        futures.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            futures.push_back(submit([&f, i] { f(i); }));
        }
        std::exception_ptr first_error;
        for (auto& fut : futures) {
            try {
                fut.get();
            } catch (...) {
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (first_error) std::rethrow_exception(first_error);
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Resolve an n_jobs-style parameter (sklearn convention: None -> 1
// (sequential); -1 -> all available hardware threads; positive int -> that
// many threads) to an actual thread count, clamped to >= 1. Callers pass 1
// for "None".
inline std::size_t resolve_n_jobs(int n_jobs) {
    if (n_jobs < 0) {
        unsigned hw = std::thread::hardware_concurrency();
        return hw > 0 ? static_cast<std::size_t>(hw) : 1;
    }
    if (n_jobs == 0) return 1;
    return static_cast<std::size_t>(n_jobs);
}

} // namespace rdt
