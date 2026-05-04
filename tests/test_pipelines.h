#pragma once
#include <voco/pipeline.h>

voco::ComputePipeline& scalePipeline();
voco::ComputePipeline& inplaceAddPipeline();
voco::ComputePipeline& uniformPipeline();
voco::ComputePipeline& multisetPipeline();
void cleanupTestPipelines();
