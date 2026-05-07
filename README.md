# voco

voco is a **thin C++ layer over Vulkan compute**. It handles descriptor management, synchronization, and resource lifetime so dispatch calls stay clean and the boilerplate stays out of your code.

voco is designed to:
- Let you write GPU compute code without managing descriptor sets, pipeline layouts, or semaphores by hand
- Fit into any Vulkan codebase without dictating instance creation, device selection, or extensions
- Be transparent enough that you can drop down to raw Vulkan whenever you need to

```cpp
voco::CommandList cmd = device.createCommandList();
cmd.bindPipeline(pipeline);
cmd.bindBuffer(0, 0, inputBuf,  voco::Access::Read);
cmd.bindBuffer(0, 1, outputBuf, voco::Access::Write);
cmd.setPushConstants(Params{ 2.0f });
cmd.dispatch(groupsX, 1, 1);
device.submit(cmd);
```

Barriers between dispatches are inserted automatically. Cross-submission hazards on the same buffer are detected and resolved with timeline semaphore waits. Descriptor sets are cached and reused across submissions.

---

## Requirements

- Vulkan 1.3+
- CMake 3.25+
- C++23

Dependencies (bundled under `external/`): 
- [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [SPIRV-Reflect](https://github.com/KhronosGroup/SPIRV-Reflect)

Optional: Vulkan SDK with `shaderc_combined` for runtime GLSL compilation (`-DVOCO_ENABLE_GLSL=ON`)

## Getting started

voco is built as a static library and can be linked into any program. The recommended way to integrate it is via CMake's `add_subdirectory`. voco can be built with any compiler that supports C++23. Clone or copy voco into your project tree, then in your `CMakeLists.txt`:

```cmake
add_subdirectory(voco)
target_link_libraries(your_target PRIVATE voco)
```

Include directories are propagated automatically through the `voco` target. The entire API is then accessible via a single include:

```cpp
#include <voco/voco.h>
```

To build voco standalone:

```bash
cmake -B build
cmake --build build
```

GLSL runtime compilation is off by default. Enable it if you have the Vulkan SDK and want to pass `.glsl` source directly:

```bash
cmake -B build -DVOCO_ENABLE_GLSL=ON
```

## Usage

voco fits into an existing Vulkan application rather than owning the setup. Instance creation, physical device selection, and extension management are yours to control. Pass your existing handles into a `voco::Context` to create a `voco::Device`:

```cpp
voco::Context ctx{};
ctx.instance                = vkInstance;
ctx.physicalDevice          = vkPhysicalDevice;
ctx.device                  = vkDevice;
ctx.computeQueue            = computeQueue;
ctx.computeQueueFamilyIndex = computeQueueFamilyIndex;

voco::Device device(ctx);
```

`voco::Device` is the central object. Everything else (buffers, pipelines, command lists) is created through it and tied to its lifetime. All voco objects follow RAII and release their Vulkan resources when they go out of scope. Let `Device` go out of scope only after all objects created from it are done.

## Buffers

`Buffer` is the primary data container for GPU compute work. It wraps a VMA allocation and records which submissions have touched it, giving voco the information it needs to insert barriers and resolve cross-submission hazards. `MemoryType` controls placement: most compute buffers belong in `Device` memory; use `Host` for staging or data you update frequently from the CPU.

```cpp
voco::Buffer buf = device.createBuffer(
    voco::BufferUsage::Storage | voco::BufferUsage::TransferDst |voco::BufferUsage::TransferSrc,
    1024 * sizeof(float));

device.copyToDevice(data.data(), buf, 0, byteSize);
device.copyToHost(buf, data.data(), 0, byteSize);

voco::Future f = device.copyToDeviceAsync(data.data(), buf, 0, byteSize);
f.wait();
```

Flags can be combined with `|` to describe all the ways the buffer will be used:

```cpp
voco::BufferUsage::Storage | voco::BufferUsage::TransferSrc | voco::BufferUsage::TransferDst
```

| `BufferUsage` | Use |
|---|---|
| `Storage` | Read/write access from shaders |
| `Uniform` | Read-only uniform buffer access from shaders |
| `TransferSrc` | Source of a copy operation (e.g. copyToHost) |
| `TransferDst` | Destination of a copy operation (e.g. copyToDevice) |

| `MemoryType` | Use |
|---|---|
| `Device` | GPU-local, fastest for compute |
| `Host` | CPU-visible, good for staging and UBOs |
| `Unified` | Shared CPU/GPU (if available) |

## Pipelines

In raw Vulkan, a compute pipeline requires manually declaring every binding and push constant range through `VkDescriptorSetLayoutBinding` and `VkPipelineLayoutCreateInfo`. `ComputePipeline` handles that automatically. voco reads the SPIR-V through spirv-reflect to build the layout, so you only need to point it at a shader file.

```cpp
// pre-compiled SPIR-V
voco::ComputePipeline pipeline = device.createComputePipeline("shader.spv");

// GLSL source (requires VOCO_ENABLE_GLSL)
voco::ComputePipeline pipeline = device.createComputePipeline("shader.glsl", voco::ShaderSourceType::GLSL);
```

## Command lists

`CommandList` records a sequence of compute operations: pipeline binds, resource binds, push constant updates, and dispatches. voco tracks buffer accesses across dispatches to insert barriers at the right points. Submit the list with `device.submit()` when you are ready to execute.

```cpp
voco::CommandList cmd = device.createCommandList();
cmd.bindPipeline(pipeline);
cmd.bindBuffer(0, 0, inputBuf,  voco::Access::Read);
cmd.bindBuffer(0, 1, outputBuf, voco::Access::Write);
cmd.setPushConstants(Params{ 2.0f });
cmd.dispatch(groupsX, 1, 1);
device.submit(cmd);
```

`Access` tells voco how a buffer will be used in a dispatch, which is the information it needs to insert correct barriers:

| `Access` | Use |
|---|---|
| `Read` | Shader reads the buffer but does not write it |
| `Write` | Shader writes the buffer but does not read it |
| `ReadWrite` | Shader both reads and writes the buffer |

## Synchronization

Barriers between dispatches on the same buffer are inserted automatically:

```cpp
cmd.bindBuffer(0, 0, buf, voco::Access::ReadWrite);
cmd.dispatch(1, 1, 1);
cmd.bindBuffer(0, 0, buf, voco::Access::ReadWrite); // barrier inserted
cmd.dispatch(1, 1, 1);
```

Cross-submission hazards are detected in `Device::submit`. If a bound buffer is still in-flight from a previous submission, a timeline semaphore wait is added before the new submission goes to the queue.

## Descriptor sets

voco manages descriptor sets automatically. When you call `bindBuffer` and then `dispatch`, voco looks up a descriptor set matching the exact combination of buffers and bindings you have bound. If one exists and the GPU has finished using it, it is reused. If not, a new one is allocated and written. Each descriptor set is written once when allocated and reused on subsequent dispatches once the GPU has finished with it. If the set is still in use by the GPU, a new one is allocated for that submission. Sets are freed when the buffers they reference are destroyed.

This means you pay no allocation or write cost on repeated dispatches with the same inputs, and you never touch `VkDescriptorPool`, `VkDescriptorSetLayout`, or `vkUpdateDescriptorSets` directly.

### Multiple descriptor sets

Sets not rebound between dispatches remain bound from the previous dispatch, following the standard Vulkan descriptor set model. A common pattern is pinning shared resources like a UBO or lookup table to set 0, then swapping only set 1 per dispatch or across pipeline switches.

```cpp
cmd.bindBuffer(0, 0, inputBuf,  voco::Access::Read);
cmd.bindBuffer(0, 1, outputBuf, voco::Access::Write);
cmd.bindBuffer(1, 0, ubo,       voco::Access::Read);
cmd.dispatch(1, 1, 1);

// set 1 stays bound
cmd.bindBuffer(0, 0, inputBuf2,  voco::Access::Read);
cmd.bindBuffer(0, 1, outputBuf2, voco::Access::Write);
cmd.dispatch(1, 1, 1);
```

## Scope

**Planned**

Features within voco's scope that are not yet implemented:

- Images and sampled images
- HLSL shader compilation via DXC
- Dedicated transfer queue for staging operations
- Async compute across multiple queues (on hardware that supports it)
- Multi-GPU compute

**Out of scope**

voco is a compute-only library. The following are intentionally not supported:

- Graphics pipelines (vertex, fragment, geometry, etc.)
- Vulkan instance and device creation
- Extension and feature management
