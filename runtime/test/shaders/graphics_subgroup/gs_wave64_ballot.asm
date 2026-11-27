; mattr: +wavefrontsize64,-wavefrontsize32
; One invocation per triangle of the vertex grid. Export the same ballot counts at each emitted vertex.
; The three position-read and emit sequences use the legacy GS ring layout in gs_tri.asm.
s_bfe_i32 s0, s3, 0x80008
s_bfm_b64 exec, s0, 0
s_cmp_eq_u32 s0, 64
s_cselect_b64 exec, -1, exec
v_mbcnt_lo_u32_b32 v30, -1, 0
v_mbcnt_hi_u32_b32 v30, -1, v30
v_cmp_ge_u32_e64 s[6:7], v30, 32
s_bcnt1_i32_b64 s8, s[6:7]
s_bcnt1_i32_b64 s9, exec
v_mov_b32_e32 v31, s8
v_mov_b32_e32 v32, s9
v_mov_b32_e32 v33, 1
v_lshlrev_b32_e32 v3, 2, v0
v_and_b32_e32 v10, 0x3fffc, v3
s_mov_b32 s1, 2
v_lshlrev_b32_sdwa v8, s1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
v_lshlrev_b32_e32 v1, 2, v1
v_and_b32_e32 v6, 0x3fffc, v1
v_mov_b32_e32 v12, 0
ds_read2_b32 v[20:21], v10 offset1:1
ds_read2_b32 v[22:23], v10 offset0:2 offset1:3
s_waitcnt lgkmcnt(0)
ds_write2_b32 v12, v20, v21 offset0:0 offset1:1
ds_write2_b32 v12, v22, v23 offset0:2 offset1:3
ds_write2_b32 v12, v30, v31 offset0:4 offset1:5
ds_write2_b32 v12, v32, v33 offset0:6 offset1:7
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_read2_b32 v[20:21], v8 offset1:1
ds_read2_b32 v[22:23], v8 offset0:2 offset1:3
s_waitcnt lgkmcnt(0)
ds_write2_b32 v12, v20, v21 offset0:8 offset1:9
ds_write2_b32 v12, v22, v23 offset0:10 offset1:11
ds_write2_b32 v12, v30, v31 offset0:12 offset1:13
ds_write2_b32 v12, v32, v33 offset0:14 offset1:15
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
ds_read2_b32 v[20:21], v6 offset1:1
ds_read2_b32 v[22:23], v6 offset0:2 offset1:3
s_waitcnt lgkmcnt(0)
ds_write2_b32 v12, v20, v21 offset0:16 offset1:17
ds_write2_b32 v12, v22, v23 offset0:18 offset1:19
ds_write2_b32 v12, v30, v31 offset0:20 offset1:21
ds_write2_b32 v12, v32, v33 offset0:22 offset1:23
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
