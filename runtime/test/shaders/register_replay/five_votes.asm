v_and_b32_e32 v13, 1, v4
v_and_b32_e32 v19, 2, v4
v_and_b32_e32 v22, 32, v4
v_and_b32_e32 v26, 12, v4
v_cmp_ne_u32_e64 s[14:15], v13, 0
s_and_saveexec_b64 s[16:17], s[14:15]
v_cmp_ne_u32_e64 s[14:15], v19, 0
s_and_saveexec_b64 s[18:19], s[14:15]
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v19, 0
v_cmp_ne_u32_e64 s[20:21], s10, v22
s_bcnt1_i32_b64 s23, s[20:21]
v_cmp_ne_u32_e64 s[20:21], s23, v26
s_bcnt1_i32_b64 s24, s[20:21]
v_add_nc_u32_e32 v4, s24, v4
s_andn2_b64 exec, s[18:19], exec
v_cmp_ne_u32_e64 s[20:21], s24, v26
s_bcnt1_i32_b64 s25, s[20:21]
v_cmp_ne_u32_e64 s[20:21], s25, v22
s_bcnt1_i32_b64 s26, s[20:21]
v_cmp_ne_u32_e64 s[20:21], s26, v26
s_bcnt1_i32_b64 s27, s[20:21]
v_add_nc_u32_e32 v5, s27, v5
s_mov_b64 exec, s[18:19]
s_mov_b64 exec, s[16:17]
v_mov_b32_e32 v27, s23
v_mov_b32_e32 v28, s24
v_mov_b32_e32 v29, s25
v_mov_b32_e32 v30, s26
v_mov_b32_e32 v31, s27
