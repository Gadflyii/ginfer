#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ginfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue, int HeadDimValue = 256>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);
    static_assert(HeadDimValue == 128 || HeadDimValue == 256);
    static_assert(HeadDimValue % 64 == 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int HeadDim          = HeadDimValue;
    static constexpr int QuantGroups      = HeadDimValue / 64;
};

inline constexpr int DecodeSplitsMax = 256;

using Gqa27Geometry   = GqaGeometry<24, 4, 1>;
using Gqa35Geometry   = GqaGeometry<16, 2, 2>;
using GqaMuseGeometry = GqaGeometry<32, 2, 2, 128>;

} // namespace ginfer::ops
