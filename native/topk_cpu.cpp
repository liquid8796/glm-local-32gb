// Bounded finite FP32 top-k compatibility experiment for the pinned CPU runtime.
// The STL selection strategy matches that used by ATen CPU TopKImpl.h:
// partial_sort for k*64<=n, otherwise nth_element followed by sorting k-1.
// Equal-value selection is deliberately library-specific, never index-stable.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

extern "C" __declspec(dllexport) int mini_topk(
    const float* values, std::uint32_t size, std::uint32_t count,
    std::int32_t* output) noexcept {
    if (!values || !output || size < 1 || size > 128 || count < 1 || count > 4 || count > size)
        return 1;
    std::array<std::pair<float, std::int32_t>, 128> entries{};
    for (std::uint32_t i = 0; i < size; ++i) {
        if (!std::isfinite(values[i])) return 2;
        entries[i] = {values[i], static_cast<std::int32_t>(i)};
    }
    const auto greater = [](const auto& a, const auto& b) { return a.first > b.first; };
    auto begin = entries.begin();
    auto end = begin + size;
    if (count * 64 <= size) {
        std::partial_sort(begin, begin + count, end, greater);
    } else {
        std::nth_element(begin, begin + count - 1, end, greater);
        std::sort(begin, begin + count - 1, greater);
    }
    for (std::uint32_t i = 0; i < count; ++i) output[i] = entries[i].second;
    return 0;
}
