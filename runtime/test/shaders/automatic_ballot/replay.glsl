#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_buffer_reference : require

// A non-indexed vertex ID or a single-sample, single-primitive pixel coordinate is unique in this fixture.
struct Lane {
    uint slot;
    uint value;
    uint finalValue;
    uint visits;
};
layout(buffer_reference, std430, buffer_reference_align=4) buffer Scratch {
    uint halves;
    uint errors;
    uint changedLane;
    uint padding;
    uint occupied[512];
    uint first[512];
    uint second[512];
    Lane lanes[1024];
};
layout(push_constant) uniform Params {
    uint phase;
    uint unused;
    uint count;
    uint rowWidth;
    uint seed;
    layout(offset=128) Scratch s;
} pc;
#ifdef VERTEX_STAGE
layout(location=1) flat out vec4 result;
#else
layout(location=0) out vec4 result;
#endif

uint valueFor(uint key) {
    uint x = key * 747796405u + pc.seed;
    return (x ^ (x >> 16)) * 2246822519u;
}
uint population(uint lo, uint hi) {
    return uint(bitCount(lo) + bitCount(hi));
}
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
    if (id >= pc.count || gl_SubgroupSize != 32u) {
        atomicAdd(pc.s.errors, 1u);
        return;
    }
    // Reorder the consumers deliberately. Physical lane IDs can change between draws.
    uint key = (id + pc.phase * 37u) % pc.count;
    if (pc.phase == 0u) {
        uint halfID = 0u;
        bool leader = subgroupElect();
        if (leader) halfID = atomicAdd(pc.s.halves, 1u);
        halfID = subgroupBroadcastFirst(halfID);
        if (halfID >= 512u) { atomicAdd(pc.s.errors, 1u); return; }
        uint value = valueFor(key);
        uint live = subgroupBallot(true).x;
        uint votes = subgroupBallot((value & 8u) != 0u).x;
        if (leader) {
            pc.s.occupied[halfID] = live;
            pc.s.first[halfID] = votes;
        }
        pc.s.lanes[key].slot = halfID * 32u + gl_SubgroupInvocationID;
        pc.s.lanes[key].value = value;
    } else {
        uint slot = pc.s.lanes[key].slot;
        if (slot >= 512u * 32u) { atomicAdd(pc.s.errors, 1u); return; }
        uint halfID = slot / 32u;
        uint pair = halfID & ~1u;
        uint value = pc.s.lanes[key].value;
        if (pc.phase == 1u) {
            uint firstCount = population(pc.s.first[pair], pc.s.first[pair + 1u]);
            value ^= (firstCount + 1u) * 0x9e3779b9u + (slot & 63u);
            pc.s.lanes[key].value = value;
            if ((value & 16u) != 0u) atomicOr(pc.s.second[halfID], 1u << (slot & 31u));
            if ((slot & 31u) != gl_SubgroupInvocationID) atomicAdd(pc.s.changedLane, 1u);
        } else {
            uint secondCount = population(pc.s.second[pair], pc.s.second[pair + 1u]);
            uint finalValue = value + secondCount * 17u;
            pc.s.lanes[key].finalValue = finalValue;
            result = vec4(finalValue & 65535u, finalValue >> 16, secondCount, slot);
        }
    }
    atomicAdd(pc.s.lanes[key].visits, 1u << (pc.phase * 4u));
}
