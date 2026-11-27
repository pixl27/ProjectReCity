; mattr: +wavefrontsize64,-wavefrontsize32
s_mov_b64 exec, -1
s_mov_b32 s3, s0
v_add_nc_u32_e32 v4, s0, v0
v_xor_b32_e32 v5, 37, v4
v_and_b32_e32 v7, 8, v5
v_cmp_ne_u32_e64 s[8:9], v7, 0
s_bcnt1_i32_b64 s10, s[8:9]
v_xor_b32_e32 v4, s10, v4
v_add_nc_u32_e32 v11, v5, v4
v_and_b32_e32 v7, 16, v11
v_cmp_ne_u32_e64 s[8:9], v7, 0
s_bcnt1_i32_b64 s12, s[8:9]
v_add_nc_u32_e32 v4, s12, v4
v_add_nc_u32_e32 v5, s3, v5
v_mov_b32_e32 v6, s10
v_mov_b32_e32 v7, s12
