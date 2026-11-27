; No else arm. The taken edge must preserve the incoming v4 and s3.
s_cmp_ge_u32 s10, 30
s_cbranch_scc1 2
v_xor_b32_e32 v4, s10, v4
s_mov_b32 s3, 9
