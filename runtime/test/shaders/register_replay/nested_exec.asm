; v18 records leaf membership, both inner restores, and the final restore.
v_mov_b32_e32 v18, 0
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
v_and_b32_e32 v13, 1, v4
v_and_b32_e32 v19, 2, v4
v_cmp_ne_u32_e64 s[14:15], v13, 0
s_and_saveexec_b64 s[16:17], s[14:15]
v_cmp_ne_u32_e64 s[14:15], v19, 0
s_and_saveexec_b64 s[18:19], s[14:15]
v_xor_b32_e32 v4, s10, v4
v_add_nc_u32_e32 v18, 1, v18
s_andn2_b64 exec, s[18:19], exec
v_add_nc_u32_e32 v5, 11, v5
v_add_nc_u32_e32 v18, 2, v18
s_mov_b64 exec, s[18:19]
v_add_nc_u32_e32 v18, 16, v18
; Save this lane's vote after restoring the parent mask, before the outer join.
v_add_nc_u32_e32 v22, v4, v5
v_and_b32_e32 v22, 16, v22
v_add_nc_u32_e32 v22, v22, v22
v_add_nc_u32_e32 v22, v22, v22
v_add_nc_u32_e32 v22, v22, v22
v_add_nc_u32_e32 v18, v22, v18
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, v5
s_andn2_b64 exec, s[16:17], exec
v_cmp_ne_u32_e64 s[14:15], v19, 0
s_and_saveexec_b64 s[18:19], s[14:15]
v_add_nc_u32_e32 v4, 5, v4
v_add_nc_u32_e32 v18, 4, v18
s_andn2_b64 exec, s[18:19], exec
v_xor_b32_e32 v5, 17, v5
v_add_nc_u32_e32 v18, 8, v18
s_mov_b64 exec, s[18:19]
v_add_nc_u32_e32 v18, 32, v18
; Save this lane's vote after restoring the parent mask, before the outer join.
v_add_nc_u32_e32 v22, v4, v5
v_and_b32_e32 v22, 16, v22
v_add_nc_u32_e32 v22, v22, v22
v_add_nc_u32_e32 v22, v22, v22
v_add_nc_u32_e32 v22, v22, v22
v_add_nc_u32_e32 v18, v22, v18
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, v5
s_mov_b64 exec, s[16:17]
v_add_nc_u32_e32 v18, 64, v18
v_add_nc_u32_e32 v4, 3, v4
