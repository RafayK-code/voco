#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Buf { float data[]; } buf;

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] += 1.0;
}
