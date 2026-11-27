; Predicate depends on invocation data. Preserve the other arm's VGPR on each lane.
v_and_b32_e32 v13, 1, v4
v_cmp_ne_u32_e64 s[14:15], v13, 0
s_and_saveexec_b64 s[16:17], s[14:15]
v_xor_b32_e32 v4, s10, v4
v_add_nc_u32_e32 v4, 1, v4
s_andn2_b64 exec, s[16:17], exec
v_add_nc_u32_e32 v5, 17, v5
s_mov_b64 exec, s[16:17]
