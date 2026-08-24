#include "ginfer/ops/grouped_dynamic_conv.h"

#include "ops/launcher/grouped_dynamic_conv.h"

#include <cstddef>
#include <stdexcept>

namespace ginfer::ops {
namespace {

void require_batch(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t width,
                   std::int32_t batch, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != width || t.ne[2] != batch ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("grouped_dynamic_conv: invalid ") + name);
    }
}

} // namespace

void grouped_dynamic_conv_prepare(const Tensor& hidden, const Tensor& proj, const Tensor& base,
                                  Tensor& dynamic, Tensor& out, std::int32_t group_size,
                                  cudaStream_t stream) {
    if (group_size <= 0 || hidden.ne[0] % group_size != 0) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid group size");
    }
    const int h = hidden.ne[0];
    const int width = hidden.ne[1];
    const int batch = hidden.ne[2];
    if (width <= 0 || width > 16 || batch <= 0 || batch > 8) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid W or B");
    }
    require_batch(hidden, DType::BF16, h, width, batch, "hidden");
    require_batch(out, DType::BF16, h, width, batch, "out");
    const int dynamic_rows = 4 * (h / group_size);
    require_batch(dynamic, DType::BF16, dynamic_rows, width, batch, "dynamic");
    if (hidden.data == dynamic.data || hidden.data == out.data || dynamic.data == out.data) {
        throw std::invalid_argument("grouped_dynamic_conv: tensors must not alias");
    }
    if (proj.dtype != DType::BF16 || proj.ne[0] != h || proj.ne[1] != dynamic_rows ||
        proj.ne[2] != 1 || proj.ne[3] != 1 || proj.data == nullptr || !proj.is_contiguous() ||
        base.dtype != DType::BF16 || base.ne[0] != h || base.ne[1] != 2 || base.ne[2] != 2 ||
        base.ne[3] != 1 || base.data == nullptr || !base.is_contiguous()) {
        throw std::invalid_argument("grouped_dynamic_conv: proj/base must be contiguous BF16");
    }
    detail::grouped_dynamic_conv_prepare_launch(hidden, proj, base, dynamic, out, group_size,
                                                stream);
}

void grouped_dynamic_conv_prepare(const Tensor& hidden, const Weight& projection,
                                  const Tensor& base, Tensor& dynamic, Tensor& out,
                                  std::int32_t group_size, cudaStream_t stream) {
    if (group_size <= 0 || hidden.ne[0] % group_size != 0) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid group size");
    }
    const int h = hidden.ne[0];
    const int width = hidden.ne[1];
    const int batch = hidden.ne[2];
    if (width <= 0 || width > 16 || batch <= 0 || batch > 8) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid W or B");
    }
    require_batch(hidden, DType::BF16, h, width, batch, "hidden");
    require_batch(out, DType::BF16, h, width, batch, "out");
    const int dynamic_rows = 4 * (h / group_size);
    require_batch(dynamic, DType::BF16, dynamic_rows, width, batch, "dynamic");
    if (hidden.data == dynamic.data || hidden.data == out.data || dynamic.data == out.data ||
        projection.qdata == hidden.data || projection.qdata == dynamic.data ||
        projection.qdata == out.data) {
        throw std::invalid_argument("grouped_dynamic_conv: tensors must not alias");
    }
    if (base.dtype != DType::BF16 || base.ne[0] != h || base.ne[1] != 2 || base.ne[2] != 2 ||
        base.ne[3] != 1 || base.data == nullptr || !base.is_contiguous()) {
        throw std::invalid_argument("grouped_dynamic_conv: base must be contiguous BF16");
    }
    if (projection.ndim != 2 || projection.n != dynamic_rows || projection.k != h ||
        projection.shape[0] != dynamic_rows || projection.shape[1] != h ||
        projection.qdata == nullptr) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid projection shape");
    }

    if (projection.qtype == QType::BF16_CTRL) {
        const std::uint64_t required =
            static_cast<std::uint64_t>(dynamic_rows) * h * sizeof(std::uint16_t);
        if (projection.layout != QuantLayout::Contiguous || projection.qhigh != nullptr ||
            projection.scales != nullptr || projection.payload_bytes < required) {
            throw std::invalid_argument("grouped_dynamic_conv: invalid BF16 projection");
        }
        Tensor dense(const_cast<void*>(projection.qdata), DType::BF16, {h, dynamic_rows});
        grouped_dynamic_conv_prepare(hidden, dense, base, dynamic, out, group_size, stream);
        return;
    }
    if (projection.qtype != QType::Q4G64_F16S && projection.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument("grouped_dynamic_conv: unsupported projection profile");
    }
    detail::grouped_dynamic_conv_prepare_launch(hidden, projection, base, dynamic, out, group_size,
                                                stream);
}

void grouped_dynamic_conv_finish(const Tensor& hidden, const Tensor& dynamic, const Tensor& base,
                                 Tensor& out, std::int32_t group_size, cudaStream_t stream) {
    if (group_size <= 0 || hidden.ne[0] % group_size != 0) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid group size");
    }
    const int h = hidden.ne[0];
    const int width = hidden.ne[1];
    const int batch = hidden.ne[2];
    if (width <= 0 || width > 16 || batch <= 0 || batch > 8) {
        throw std::invalid_argument("grouped_dynamic_conv: invalid W or B");
    }
    require_batch(hidden, DType::BF16, h, width, batch, "hidden");
    require_batch(out, DType::BF16, h, width, batch, "out");
    const int dynamic_rows = 4 * (h / group_size);
    require_batch(dynamic, DType::BF16, dynamic_rows, width, batch, "dynamic");
    if (hidden.data == dynamic.data || hidden.data == out.data || dynamic.data == out.data) {
        throw std::invalid_argument("grouped_dynamic_conv: tensors must not alias");
    }
    if (base.dtype != DType::BF16 || base.ne[0] != h || base.ne[1] != 2 || base.ne[2] != 2 ||
        base.ne[3] != 1 || base.data == nullptr || !base.is_contiguous()) {
        throw std::invalid_argument("grouped_dynamic_conv: dynamic/base must be contiguous BF16");
    }
    detail::grouped_dynamic_conv_finish_launch(hidden, dynamic, base, out, group_size, stream);
}

} // namespace ginfer::ops
