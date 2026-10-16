#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

// Seedable PRNG helpers for the ensemble layer: bootstrap resampling,
// random row/column subset selection, and deterministic per-tree/per-round
// seed derivation so parallel building stays reproducible under a fixed
// random_state regardless of thread-scheduling order.
namespace rdt {

class Rng {
public:
    explicit Rng(std::uint64_t seed) : engine_(seed) {}

    std::mt19937_64& engine() { return engine_; }

    // n indices in [0, population) with replacement -- bootstrap resampling.
    std::vector<std::uint32_t> sample_with_replacement(std::uint32_t population, std::uint32_t n) {
        std::uniform_int_distribution<std::uint32_t> dist(0, population - 1);
        std::vector<std::uint32_t> out(n);
        for (std::uint32_t i = 0; i < n; ++i) out[i] = dist(engine_);
        return out;
    }

    // n distinct indices in [0, population) without replacement, ascending
    // -- row/feature subsampling (subsample, max_features/colsample_bytree).
    std::vector<std::uint32_t> sample_without_replacement(std::uint32_t population, std::uint32_t n) {
        n = std::min(n, population);
        std::vector<std::uint32_t> pool(population);
        for (std::uint32_t i = 0; i < population; ++i) pool[i] = i;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uniform_int_distribution<std::uint32_t> dist(i, population - 1);
            std::uint32_t j = dist(engine_);
            std::swap(pool[i], pool[j]);
        }
        // The first n entries in ascending order: for a large sample, mark
        // them and read them back in order (linear) instead of sorting.
        if (n >= population / 16) {
            std::vector<char> chosen(population, 0);
            for (std::uint32_t i = 0; i < n; ++i) chosen[pool[i]] = 1;
            pool.clear();
            for (std::uint32_t v = 0; v < population; ++v) {
                if (chosen[v]) pool.push_back(v);
            }
            return pool;
        }
        pool.resize(n);
        std::sort(pool.begin(), pool.end());
        return pool;
    }

private:
    std::mt19937_64 engine_;
};

// splitmix64-style mixing: cheap, well-distributed for the small index
// ranges (n_estimators, n_classes) this is used for. Same master_seed always
// derives the same per-index seed, independent of build/thread order.
inline std::uint64_t derive_seed(std::uint64_t master_seed, std::size_t index) {
    std::uint64_t z = master_seed + 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(index) + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

} // namespace rdt
