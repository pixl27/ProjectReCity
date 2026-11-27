; Outer condition uses the first ballot count. Both arms contain a second SCC branch on the seed.
s_cmp_ge_u32 s10, 30
s_cbranch_scc0 6
s_cmp_eq_u32 s0, 7
s_cbranch_scc0 2
s_mov_b32 s3, 9
s_branch 1
v_xor_b32_e32 v4, s10, v4
s_branch 5
s_cmp_eq_u32 s0, 7
s_cbranch_scc1 2
v_add_nc_u32_e32 v5, 5, v5
s_branch 1
s_mov_b32 s3, 17
