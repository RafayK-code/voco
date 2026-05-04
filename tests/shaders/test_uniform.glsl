#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer InputBuf { float data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;
layout(set = 0, binding = 2) uniform Params { float scale; float bias; } params;

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    outBuf.data[idx] = inBuf.data[idx] * params.scale + params.bias;
}
