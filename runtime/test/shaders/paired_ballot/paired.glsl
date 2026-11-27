// Synthetic graphics ballot with two capture draws and one readback draw. Both stages use the same code.
#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_buffer_reference : require
layout(buffer_reference, std430, buffer_reference_align=4) buffer Votes {
    uint inputs[64];
    uint width[2];
    uint ballot[2];
};
layout(push_constant) uniform Params {
    uint phase;
    uint reference64;
    layout(offset=128) Votes votes;
} pc;
#ifdef VERTEX_STAGE
layout(location=1) flat out vec4 result;
#else
layout(location=0) out vec4 result;
#endif

void main() {
#ifdef VERTEX_STAGE
    const vec2 triangle[] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(triangle[gl_VertexIndex % 3], 0, 1);
#endif
    result = vec4(-1);
    uint expectedWidth = pc.reference64 != 0 ? 64u : 32u;
    if (pc.phase < 2u) {
        if (pc.reference64 != 0 && pc.phase != 0u) return;
        uint logicalLane = (gl_SubgroupInvocationID + pc.phase * 32u) & 63u;
        uvec4 mask = subgroupBallot(pc.votes.inputs[logicalLane] != 0u);
        uint activeCount = subgroupBallotBitCount(subgroupBallot(true));
        bool leader = subgroupElect();
        // All full subgroups in this draw evaluate the same logical half. Keep exactly one.
        if (gl_SubgroupSize == expectedWidth && activeCount == expectedWidth && leader) {
            if (atomicCompSwap(pc.votes.width[pc.phase], 0u, gl_SubgroupSize) == 0u) {
                pc.votes.ballot[pc.phase] = mask.x;
                if (pc.reference64 != 0) {
                    pc.votes.width[1] = gl_SubgroupSize;
                    pc.votes.ballot[1] = mask.y;
                }
            }
        }
    } else if (gl_SubgroupSize == expectedWidth &&
               pc.votes.width[0] == expectedWidth && pc.votes.width[1] == expectedWidth) {
        uint lo = pc.votes.ballot[0], hi = pc.votes.ballot[1];
        uint count = uint(bitCount(lo) + bitCount(hi));
        // Exact integers in an RGBA32F attachment, including for an all-ones ballot.
        result = vec4(lo & 65535u, lo >> 16, hi & 65535u, (hi >> 16) + count * 65536u);
    }
}
