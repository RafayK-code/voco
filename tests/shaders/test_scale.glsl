#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer InputBuf { float data[]; } inBuf;
layout(set = 0, binding = 1) buffer OutputBuf { float data[]; } outBuf;

layout(push_constant) uniform PC { float scale; } pc;

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    outBuf.data[idx] = inBuf.data[idx] * pc.scale;
}
