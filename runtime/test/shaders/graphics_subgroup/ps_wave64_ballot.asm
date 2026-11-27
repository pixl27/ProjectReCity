; mattr: +wavefrontsize64,-wavefrontsize32
; Full-screen triangle. Export (lane, upper votes, total votes, marker).
; Expected counts with 64 active lanes: (32,64). A full physical wave32 observes (0,32).
s_mov_b64 exec, -1
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
v_cmp_ge_u32_e64 s[6:7], v0, 32
s_bcnt1_i32_b64 s8, s[6:7]
s_bcnt1_i32_b64 s9, exec
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, s9
v_mov_b32_e32 v3, 1
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
