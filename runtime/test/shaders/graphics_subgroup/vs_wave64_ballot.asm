; mattr: +wavefrontsize64,-wavefrontsize32
; Draw 6 vertices per instance across 64 instances. Each pair of triangles covers one tile in an 8 by 8 grid.
; The flat varying reports (lane, upper-half count, active count, marker) to a readback pixel shader.
; With all 64 lanes active the two counts are (32,64). A full native wave32 observes (0,32).
s_mov_b64 exec, -1
v_mov_b32_e32 v4, 0x32
v_lshrrev_b32_e32 v4, v0, v4
v_and_b32_e32 v4, 1, v4
v_mov_b32_e32 v5, 0x2c
v_lshrrev_b32_e32 v5, v0, v5
v_and_b32_e32 v5, 1, v5
v_and_b32_e32 v6, 7, v3
v_lshrrev_b32_e32 v7, 3, v3
v_add_nc_u32_e32 v4, v4, v6
v_add_nc_u32_e32 v5, v5, v7
v_cvt_f32_u32_e32 v4, v4
v_cvt_f32_u32_e32 v5, v5
v_fma_f32 v4, v4, 0.25, -1.0
v_fma_f32 v5, v5, 0.25, -1.0
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
exp pos0 v4, v5, v6, v7 done
v_mbcnt_lo_u32_b32 v8, -1, 0
v_mbcnt_hi_u32_b32 v8, -1, v8
v_cmp_ge_u32_e64 s[6:7], v8, 32
s_bcnt1_i32_b64 s8, s[6:7]
s_bcnt1_i32_b64 s9, exec
v_mov_b32_e32 v9, s8
v_mov_b32_e32 v10, s9
v_mov_b32_e32 v11, 1
exp param1 v8, v9, v10, v11
s_endpgm
