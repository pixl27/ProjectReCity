; Full EXEC. If firstCount >= 30, update v4 and s3. Otherwise only update v5.
s_cmp_ge_u32 s10, 30
s_cbranch_scc0 3
v_xor_b32_e32 v4, s10, v4
s_mov_b32 s3, 9
s_branch 1
v_add_nc_u32_e32 v5, 5, v5
