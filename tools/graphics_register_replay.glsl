#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_buffer_reference : require
layout(buffer_reference, std430, buffer_reference_align=4) buffer Scratch {
    uint halves, errors;
    uint votes[2][512];
    uint data[];
};
layout(push_constant) uniform Params {
    uint phase, unused, count, rowWidth, seed;
    layout(offset=128) Scratch s;
} pc;
#ifdef VERTEX_STAGE
layout(location=1) flat out vec4 result;
#else
layout(location=0) out vec4 result;
#endif
void main() {
#ifdef VERTEX_STAGE
    const vec2 tri[] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(tri[gl_VertexIndex % 3], 0, 1);
    uint id = uint(gl_VertexIndex);
#else
    uint id = uint(gl_FragCoord.y) * pc.rowWidth + uint(gl_FragCoord.x);
    if (gl_HelperInvocation) return;
#endif
    result = vec4(-1);
    if (gl_SubgroupSize != 32u || id >= pc.count || pc.count > 15358u / STRIDE) {
        atomicAdd(pc.s.errors, 1u);
        return;
    }
    uint key = (id + pc.phase * 37u) % pc.count;
    uint base = key * STRIDE;
    uint slot;
    if (pc.phase == 0u) {
        uint halfID = 0u;
        if (subgroupElect()) halfID = atomicAdd(pc.s.halves, 1u);
        halfID = subgroupBroadcastFirst(halfID);
        if (halfID >= 512u) { atomicAdd(pc.s.errors, 1u); return; }
        slot = 32u * halfID + gl_SubgroupInvocationID;
        pc.s.data[base] = slot;
    } else {
        slot = pc.s.data[base];
        if (slot >= 16384u) { atomicAdd(pc.s.errors, 1u); return; }
    }
    uint halfID = slot >> 5, pair = halfID & ~1u;
    atomicAdd(pc.s.data[base+1u], 1u << (4u * pc.phase));
    // GENERATED_BODY
}
