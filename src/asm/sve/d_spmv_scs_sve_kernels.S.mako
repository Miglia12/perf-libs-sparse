/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

<%namespace file="assembly_boilerplate.py.mako" import="prologue, epilogue"/>
// Input arguments
nchunks .req x0
vals .req x1
col_indx .req x2
x .req x3
y .req x4
cs .req x5
cl .req x6
row_permd2in .req x7
alphabeta .req z1

// Local
work0 .req z2
work1 .req z3
work2 .req z4
work3 .req z5
work4 .req z6
work5 .req z7
work6 .req z8
work7 .req z9
vals0 .req z10
vals1 .req z11
vals2 .req z12
vals3 .req z13
vals4 .req z14
vals5 .req z15
vals6 .req z16
vals7 .req z17
xv0 .req z18
yv0 .req z18
xv1 .req z19
yv1 .req z19
xv2 .req z20
yv2 .req z20
xv3 .req z21
yv3 .req z21
xv4 .req z22
yv4 .req z22
xv5 .req z23
yv5 .req z23
xv6 .req z24
yv6 .req z24
xv7 .req z25
yv7 .req z25
xv8 .req z26
yv8 .req z26

xcols .req z31
yrows .req z31

dat_index .req x8
col_indx_min .req x8
C .req x8
row_permd2in_it .req x8
vals_it .req x9
col_indx_it .req x10
ri .req x11

bytes .req x28

%for sigma in (True, False):
%for C in range(1, 9):
%if sigma:
	<% func_name = "d_spmv_scs_kernel_sve_C" + str(C) %>
%else:
	<% func_name = "d_spmv_scs_kernel_sve_C" + str(C) + "_sig0" %>
%endif
	${prologue(func_name)}
	stp x29, x30, [sp, #-256]!
	mov x29, sp
	stp x19, x20, [sp, #16]
	stp x21, x22, [sp, #32]
	stp x23, x24, [sp, #48]
	stp x25, x26, [sp, #64]
	stp x27, x28, [sp, #80]
	stp d8, d9, [sp, #96]
	stp d10, d11, [sp, #112]
	stp d12, d13, [sp, #128]
	stp d14, d15, [sp, #144]

	ptrue p0.b

	mov v0.d[1], v1.d[0]
	mov alphabeta.q, q0

	cmp nchunks, xzr
	beq .Lend${C}${sigma}

%if sigma:
	mov C, xzr
	incd C, all, mul #${C}
	sub row_permd2in, row_permd2in, C, lsl 3 // Undo initial addition below
%endif

.Lchunk_loop${C}${sigma}:

	sub nchunks, nchunks, 1

%for ci in range(C):
	fmov work${ci}.d, #0
%endfor

	ldr dat_index, [cs], 8

	add vals_it, vals, dat_index, lsl 3

	ldr bytes, [sp, #256]
	mul dat_index, dat_index, bytes
	add col_indx_it, col_indx, dat_index

	ldr col_indx_min, [sp, #264]
	ldr col_indx_min, [col_indx_min]
	add x, x, col_indx_min, lsl 3

	ldr ri, [cl], 8
	// If the row length is zero then just apply beta to y
	cmp ri, xzr
	beq .Lalpha_and_beta${C}${sigma}

	cmp bytes, 1
	beq .Lrow_loop${C}${sigma}_1
	cmp bytes, 2
	beq .Lrow_loop${C}${sigma}_2
	cmp bytes, 4
	beq .Lrow_loop${C}${sigma}_4
	cmp bytes, 8
	beq .Lrow_loop${C}${sigma}_8




.Lrow_loop${C}${sigma}_1:

	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 64]
	prfm pldl1keep, [x, 256]

%for ci in range(0, C):
%if ci == C/2:
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 64]
	prfm pldl1keep, [x, 256]
%endif

	ld1d { vals${ci}.d }, p0/z, [vals_it]
	incb vals_it
	ld1b { xcols.d }, p0/z, [col_indx_it]
	incd col_indx_it
	ld1d { xv${ci}.d }, p0/z, [x, xcols.d, lsl #3]
	fmla work${ci}.d, p0/m, vals${ci}.d, xv${ci}.d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_1



.Lrow_loop${C}${sigma}_2:
	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 128]
	prfm pldl1keep, [x, 256]

%for ci in range(0, C):
%if ci > 0 and ci == C/2:
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 128]
	prfm pldl1keep, [x, 256]
%endif

	ld1d { vals${ci}.d }, p0/z, [vals_it]
	incb vals_it
	ld1h { xcols.d }, p0/z, [col_indx_it]
	incw col_indx_it
	ld1d { xv${ci}.d }, p0/z, [x, xcols.d, lsl #3]
	fmla work${ci}.d, p0/m, vals${ci}.d, xv${ci}.d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_2



.Lrow_loop${C}${sigma}_4:
	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 256]
	prfm pldl1keep, [x, 256]

%for ci in range(0, C):
%if ci > 0 and ci == C/2:
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 256]
	prfm pldl1keep, [x, 256]
%endif

	ld1d { vals${ci}.d }, p0/z, [vals_it]
	incb vals_it
	ld1w { xcols.d }, p0/z, [col_indx_it]
	inch col_indx_it
	ld1d { xv${ci}.d }, p0/z, [x, xcols.d, lsl #3]
	fmla work${ci}.d, p0/m, vals${ci}.d, xv${ci}.d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_4



.Lrow_loop${C}${sigma}_8:

	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 512]
	prfm pldl1keep, [x, 256]

%for ci in range(0, C):
%if ci > 0 and ci == C/2:
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 512]
	prfm pldl1keep, [x, 256]
%endif

	ld1d { vals${ci}.d }, p0/z, [vals_it]
	incb vals_it
	ld1d { xcols.d }, p0/z, [col_indx_it]
	incb col_indx_it
	ld1d {xv${ci}.d}, p0/z, [x, xcols.d, lsl #3]
	fmla work${ci}.d, p0/m, vals${ci}.d, xv${ci}.d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_8






	// Test alpha and beta here and branch appropriately
.Lalpha_and_beta${C}${sigma}:

	// First some housekeeping...
	// reset x
	sub x, x, col_indx_min, lsl 3
	// increment col_indx_min array ptr to point to next chunk and save to stack
	ldr col_indx_min, [sp, #264]
	add col_indx_min, col_indx_min, 8
	str col_indx_min, [sp, #264]

%if sigma:
	mov C, xzr
	incd C, all, mul #${C}
	add row_permd2in, row_permd2in, C, lsl 3
	mov row_permd2in_it, row_permd2in
%endif

	// now compare and branch on alpha 'n' beta
	fmov d0, 1.0
	fcmp d1, d0
	beq .Lbeta${C}${sigma}
//alpha is not 1
	mov d0, v1.d[1]
	fcmp d0, 0.0
	beq .Lalpha${C}${sigma}
//beta is not 0

%for ci in range(0, C):
	fmul work${ci}.d, work${ci}.d, alphabeta.d[0]    //A${ci}
%if sigma:
	ld1d { yrows.d }, p0/z, [row_permd2in_it]
	incb row_permd2in_it
	ld1d { yv${ci}.d}, p0/z, [y, yrows.d, lsl #3]                 //C${ci}
	fmla work${ci}.d, yv${ci}.d, alphabeta.d[1]       //D${ci}
	st1d { work${ci}.d }, p0, [y, yrows.d, lsl #3]               //E${ci}
%else:
	ld1d { yv${ci}.d }, p0/z, [y]
	fmla work${ci}.d, yv${ci}.d, alphabeta.d[1]       //D${ci}
	st1d { work${ci}.d }, p0, [y]
	incb y
%endif
%endfor

	b .Lcontinue${C}${sigma}


// alpha is 1
.Lbeta${C}${sigma}:
	mov d0, v1.d[1]
	fcmp d0, 0.0
	beq .Lno_alpha_or_beta${C}${sigma}
//beta is not 0

%for ci in range(0, C):
%if sigma:
	ld1d { yrows.d }, p0/z, [row_permd2in_it]
	incb row_permd2in_it
	ld1d { yv${ci}.d}, p0/z, [y, yrows.d, lsl #3]                 //C${ci}
	fmla work${ci}.d, yv${ci}.d, alphabeta.d[1]       //D${ci}
	st1d { work${ci}.d }, p0, [y, yrows.d, lsl #3]               //E${ci}
%else:
	ld1d { yv${ci}.d }, p0/z, [y]
	fmla work${ci}.d, yv${ci}.d,  alphabeta.d[1]       //D${ci}
	st1d { work${ci}.d }, p0, [y]
	incb y
%endif

%endfor

	b .Lcontinue${C}${sigma}


// beta is 0
.Lalpha${C}${sigma}:
	fmov d0, 1.0
	fcmp d1, d0
	beq .Lno_alpha_or_beta${C}${sigma}
// alpha is not 1
%for ci in range(0, C):
	fmul work${ci}.d, work${ci}.d, alphabeta.d[0]    //A${ci}
%if sigma:
	ld1d { yrows.d }, p0/z, [row_permd2in_it]
	incb row_permd2in_it
	st1d { work${ci}.d }, p0, [y, yrows.d, lsl #3]               //E${ci}
%else:
	st1d { work${ci}.d }, p0, [y]
	incb y
%endif

%endfor

	b .Lcontinue${C}${sigma}


.Lno_alpha_or_beta${C}${sigma}:
%for ci in range(0, C):
%if sigma:
	ld1d { yrows.d }, p0/z, [row_permd2in_it]
	incb row_permd2in_it
	st1d { work${ci}.d }, p0, [y, yrows.d, lsl #3]               //E${ci}
%else:
	st1d { work${ci}.d }, p0, [y]
	incb y
%endif

%endfor

.Lcontinue${C}${sigma}:
	cmp nchunks, xzr
	beq .Lend${C}${sigma}
	b .Lchunk_loop${C}${sigma}


.Lend${C}${sigma}:
	ldp x19, x20, [sp, #16]
	ldp x21, x22, [sp, #32]
	ldp x23, x24, [sp, #48]
	ldp x25, x26, [sp, #64]
	ldp x27, x28, [sp, #80]
	ldp d8, d9, [sp, #96]
	ldp d10, d11, [sp, #112]
	ldp d12, d13, [sp, #128]
	ldp d14, d15, [sp, #144]
	ldp x29, x30, [sp], #256
	ret

	${epilogue(func_name)}

%endfor
%endfor
