v_and_b32_e32 v13, 1, v4
v_and_b32_e32 v19, 2, v4
v_and_b32_e32 v22, 4, v4
v_cmp_ne_u32_e64 s[14:15], v13, 0
s_and_saveexec_b64 s[16:17], s[14:15]
v_add_nc_u32_e32 v5, 3, v5
s_andn2_b64 exec, s[16:17], exec
v_cmp_ne_u32_e64 s[14:15], v19, 0
s_and_saveexec_b64 s[18:19], s[14:15]
v_add_nc_u32_e32 v5, 7, v5
s_andn2_b64 exec, s[18:19], exec
v_mov_b32_e32 v13, 1
v_mov_b32_e32 v19, 1
v_cmp_ne_u32_e64 s[20:21], v22, 0
s_bcnt1_i32_b64 s23, s[20:21]
v_add_nc_u32_e32 v4, s23, v4
s_mov_b64 exec, s[18:19]
s_mov_b64 exec, s[16:17]
