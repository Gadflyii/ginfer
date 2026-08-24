#pragma once

#include "ginfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ginfer/targets/muse_glimmer_30b/package.h>
#include <ginfer/targets/qwen3_6_27b/package.h>
#include <ginfer/targets/qwen3_6_35b_a3b/package.h>

#include <memory>
#include <variant>

namespace ginfer {

struct DeviceContext;

namespace targets {

using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
using MuseGlimmer30B = muse_glimmer_30b::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};

struct LoadedMuseGlimmer30B {
    std::unique_ptr<MuseGlimmer30B::LoadedModel> model;
    MuseGlimmer30B::Frontend frontend;

    LoadedMuseGlimmer30B(std::unique_ptr<MuseGlimmer30B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedMuseGlimmer30B();

    LoadedMuseGlimmer30B(const LoadedMuseGlimmer30B&)            = delete;
    LoadedMuseGlimmer30B& operator=(const LoadedMuseGlimmer30B&) = delete;
};

struct MuseGlimmer30BInstance {
    using Package = MuseGlimmer30B;

    std::unique_ptr<LoadedMuseGlimmer30B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<MuseGlimmer30B::Program> program;

    MuseGlimmer30BInstance(std::unique_ptr<LoadedMuseGlimmer30B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           MuseGlimmer30B::SequencePlan sequence_plan, DeviceContext& device);
    ~MuseGlimmer30BInstance();

    MuseGlimmer30BInstance(const MuseGlimmer30BInstance&)            = delete;
    MuseGlimmer30BInstance& operator=(const MuseGlimmer30BInstance&) = delete;
};

using ActiveTarget =
    std::variant<std::unique_ptr<Qwen3_6_27BInstance>, std::unique_ptr<Qwen3_6_35BA3BInstance>,
                 std::unique_ptr<MuseGlimmer30BInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ginfer
