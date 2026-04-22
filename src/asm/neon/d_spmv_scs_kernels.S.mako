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
alpha .req v0
dalpha .req d0
beta .req v1
dbeta .req d1

// Local
work0 .req v2
work1 .req v3
work2 .req v4
work3 .req v5
work4 .req v6
work5 .req v7
work6 .req v8
work7 .req v9
vals0 .req v10
vals0q .req q10
vals1 .req v11
vals1q .req q11
vals2 .req v12
vals2q .req q12
vals3 .req v13
vals3q .req q13
vals4 .req v14
vals4q .req q14
vals5 .req v15
vals5q .req q15
vals6 .req v16
vals6q .req q16
vals7 .req v17
vals7q .req q17
xv0 .req v18
yv0 .req v18
xv1 .req v19
yv1 .req v19
xv2 .req v20
yv2 .req v20
xv3 .req v21
yv3 .req v21
xv4 .req v22
yv4 .req v22
xv5 .req v23
yv5 .req v23
xv6 .req v24
yv6 .req v24
xv7 .req v25
yv7 .req v25
xv8 .req v26
yv8 .req v26
xd0 .req d18
yd0 .req d18
xd1 .req d19
yd1 .req d19
xd2 .req d20
yd2 .req d20
xd3 .req d21
yd3 .req d21
xd4 .req d22
yd4 .req d22
xd5 .req d23
yd5 .req d23
xd6 .req d24
yd6 .req d24
xd7 .req d25
yd7 .req d25
xd8 .req d26
yd8 .req d26


dat_index .req x8
col_indx_min .req x8
C .req x8
row_permd2in_it .req x8
vals_it .req x9
col_indx_it .req x10
ri .req x11

xcol0 .req x12
yrow0 .req x12
xcol1 .req x13
yrow1 .req x13
xcol2 .req x14
yrow2 .req x14
xcol3 .req x15
yrow3 .req x15
xcol4 .req x16
yrow4 .req x16
xcol5 .req x17
yrow5 .req x17
xcol6 .req x19
yrow6 .req x19
xcol7 .req x20
yrow7 .req x20
xcol8 .req x21
yrow8 .req x21
xcol9 .req x22
yrow9 .req x22
xcol10 .req x23
yrow10 .req x23
xcol11 .req x24
yrow11 .req x24
xcol12 .req x25
yrow12 .req x25
xcol13 .req x26
yrow13 .req x26
xcol14 .req x27
yrow14 .req x27
xcol15 .req x28
yrow15 .req x28

wxcol0 .req w12
wxcol1 .req w13
wxcol2 .req w14
wxcol3 .req w15
wxcol4 .req w16
wxcol5 .req w17
wxcol6 .req w19
wxcol7 .req w20
wxcol8 .req w21
wxcol9 .req w22
wxcol10 .req w23
wxcol11 .req w24
wxcol12 .req w25
wxcol13 .req w26
wxcol14 .req w27
wxcol15 .req w28

y1 .req x30
bytes .req x30

%for sigma in (True, False):
%for C in range(2, 18, 2):
%if sigma:
	<% func_name = "d_spmv_scs_kernel_neon_C" + str(C) %>
%else:
	<% func_name = "d_spmv_scs_kernel_neon_C" + str(C) + "_sig0" %>
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

	cmp nchunks, xzr
	beq .Lend${C}${sigma}

%if sigma:
	mov C, ${C}
	sub row_permd2in, row_permd2in, C, lsl 3 // Undo initial addition below
%endif

.Lchunk_loop${C}${sigma}:

	sub nchunks, nchunks, 1

%for ci in range(0, int(C/2)):
	dup work${ci}.2d, xzr
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

%for ci in range(0, int(C/2)):
%if ci == int(C/4):
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 64]
	prfm pldl1keep, [x, 256]
%endif

	ldr vals${ci}q, [vals_it], 16
	ldrb wxcol${2*ci}, [col_indx_it], 1
	ldrb wxcol${2*ci + 1}, [col_indx_it], 1
	add xcol${2*ci}, x, xcol${2*ci}, lsl 3
	add xcol${2*ci + 1}, x, xcol${2*ci + 1}, lsl 3
	ldr xd${ci}, [xcol${2*ci}]
	ld1 {xv${ci}.d}[1], [xcol${2*ci + 1}]
	fmla work${ci}.2d, vals${ci}.2d, xv${ci}.2d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_1




.Lrow_loop${C}${sigma}_2:

	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 128]
	prfm pldl1keep, [x, 256]

%for ci in range(0, int(C/2)):
%if ci == int(C/4):
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 128]
	prfm pldl1keep, [x, 256]
%endif

	ldr vals${ci}q, [vals_it], 16
	ldrh wxcol${2*ci}, [col_indx_it], 2
	ldrh wxcol${2*ci + 1}, [col_indx_it], 2
	add xcol${2*ci}, x, xcol${2*ci}, lsl 3
	add xcol${2*ci + 1}, x, xcol${2*ci + 1}, lsl 3
	ldr xd${ci}, [xcol${2*ci}]
	ld1 {xv${ci}.d}[1], [xcol${2*ci + 1}]
	fmla work${ci}.2d, vals${ci}.2d, xv${ci}.2d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_2




.Lrow_loop${C}${sigma}_4:

	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 256]
	prfm pldl1keep, [x, 256]

%for ci in range(0, int(C/2)):
%if ci == int(C/4):
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 256]
	prfm pldl1keep, [x, 256]
%endif

	ldr vals${ci}q, [vals_it], 16
	ldp wxcol${2*ci}, wxcol${2*ci + 1}, [col_indx_it], 8
	add xcol${2*ci}, x, xcol${2*ci}, lsl 3
	add xcol${2*ci + 1}, x, xcol${2*ci + 1}, lsl 3
	ldr xd${ci}, [xcol${2*ci}]
	ld1 {xv${ci}.d}[1], [xcol${2*ci + 1}]
	fmla work${ci}.2d, vals${ci}.2d, xv${ci}.2d

%endfor

	beq .Lalpha_and_beta${C}${sigma}
	b .Lrow_loop${C}${sigma}_4




.Lrow_loop${C}${sigma}_8:

	subs ri, ri, 1

	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 512]
	prfm pldl1keep, [x, 256]

%for ci in range(0, int(C/2)):
%if ci == int(C/4):
	prfm pldl1keep, [vals_it, 256]
	prfm pldl1keep, [col_indx_it, 512]
	prfm pldl1keep, [x, 256]
%endif

	ldr vals${ci}q, [vals_it], 16
	ldp xcol${2*ci}, xcol${2*ci + 1}, [col_indx_it], 16
	add xcol${2*ci}, x, xcol${2*ci}, lsl 3
	add xcol${2*ci + 1}, x, xcol${2*ci + 1}, lsl 3
	ldr xd${ci}, [xcol${2*ci}]
	ld1 {xv${ci}.d}[1], [xcol${2*ci + 1}]
	fmla work${ci}.2d, vals${ci}.2d, xv${ci}.2d

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
	mov C, ${C}
	add row_permd2in, row_permd2in, C, lsl 3
	mov row_permd2in_it, row_permd2in
%else:
	add y1, y, 8
%endif

	// now compare and branch on alpha 'n' beta
	fmov d27, 1.0
	fcmp dalpha, d27
	beq .Lbeta${C}${sigma}
//alpha is not 1
	fcmp dbeta, 0.0
	beq .Lalpha${C}${sigma}
//beta is not 0

%for ci in range(0, int(C/2)):
	fmul work${ci}.2d, work${ci}.2d, alpha.d[0]    //A${ci}
%if sigma:
	ldp yrow${2*ci}, yrow${2*ci + 1}, [row_permd2in_it], 16 //A${ci}
	add yrow${2*ci}, y, yrow${2*ci}, lsl 3              //B${ci}
	add yrow${2*ci + 1}, y, yrow${2*ci + 1}, lsl 3              //B${ci}
	ldr yd${ci}, [yrow${2*ci}]                 //C${ci}
	ld1 {yv${ci}.d}[1], [yrow${2*ci + 1}]                 //C${ci}
	fmla work${ci}.2d, yv${ci}.2d, beta.d[0]       //D${ci}
	st1 {work${ci}.d}[0], [yrow${2*ci}]               //E${ci}
	st1 {work${ci}.d}[1], [yrow${2*ci + 1}]               //E${ci}
%else:
	ldr yd${ci}, [y]                 //C${ci}
	ld1 {yv${ci}.d}[1], [y1]                 //C${ci}
	fmla work${ci}.2d, yv${ci}.2d, beta.d[0]       //D${ci}
	st1 {work${ci}.d}[0], [y]               //E${ci}
	add y, y, 16
	st1 {work${ci}.d}[1], [y1]               //E${ci}
	add y1, y1, 16
%endif
%endfor

	b .Lcontinue${C}${sigma}


// alpha is 1
.Lbeta${C}${sigma}:
	fcmp dbeta, 0.0
	beq .Lno_alpha_or_beta${C}${sigma}
//beta is not 0

%for ci in range(0, int(C/2)):
%if sigma:
	ldp yrow${2*ci}, yrow${2*ci + 1}, [row_permd2in_it], 16 //A${ci}
	add yrow${2*ci}, y, yrow${2*ci}, lsl 3              //B${ci}
	add yrow${2*ci + 1}, y, yrow${2*ci + 1}, lsl 3              //B${ci}
	ldr yd${ci}, [yrow${2*ci}]                 //C${ci}
	ld1 {yv${ci}.d}[1], [yrow${2*ci + 1}]                 //C${ci}
	fmla work${ci}.2d, yv${ci}.2d, beta.d[0]
	st1 {work${ci}.d}[0], [yrow${2*ci}]               //E${ci}
	st1 {work${ci}.d}[1], [yrow${2*ci + 1}]               //E${ci}
%else:
	ldr yd${ci}, [y]                 //C${ci}
	ld1 {yv${ci}.d}[1], [y1]                 //C${ci}
	fmla work${ci}.2d, yv${ci}.2d, beta.d[0]       //D${ci}
	st1 {work${ci}.d}[0], [y]               //E${ci}
	add y, y, 16
	st1 {work${ci}.d}[1], [y1]               //E${ci}
	add y1, y1, 16
%endif

%endfor

	b .Lcontinue${C}${sigma}


// beta is 0
.Lalpha${C}${sigma}:
	fmov d27, 1.0
	fcmp dalpha, d27
	beq .Lno_alpha_or_beta${C}${sigma}
// alpha is not 1
%for ci in range(0, int(C/2)):
	fmul work${ci}.2d, work${ci}.2d, alpha.d[0]    //A${ci}
%if sigma:
	ldp yrow${2*ci}, yrow${2*ci + 1}, [row_permd2in_it], 16 //A${ci}
	add yrow${2*ci}, y, yrow${2*ci}, lsl 3              //B${ci}
	add yrow${2*ci + 1}, y, yrow${2*ci + 1}, lsl 3              //B${ci}
	st1 {work${ci}.d}[0], [yrow${2*ci}]               //C${ci}
	st1 {work${ci}.d}[1], [yrow${2*ci + 1}]               //C${ci}
%else:
	st1 {work${ci}.d}[0], [y]               //C${ci}
	add y, y, 16
	st1 {work${ci}.d}[1], [y1]               //C${ci}
	add y1, y1, 16
%endif

%endfor

	b .Lcontinue${C}${sigma}


.Lno_alpha_or_beta${C}${sigma}:
%for ci in range(0, int(C/2)):
%if sigma:
	ldp yrow${2*ci}, yrow${2*ci + 1}, [row_permd2in_it], 16 //A${ci}
	add yrow${2*ci}, y, yrow${2*ci}, lsl 3              //B${ci}
	add yrow${2*ci + 1}, y, yrow${2*ci + 1}, lsl 3              //B${ci}
	st1 {work${ci}.d}[0], [yrow${2*ci}]               //C${ci}
	st1 {work${ci}.d}[1], [yrow${2*ci + 1}]               //C${ci}
%else:
	st1 {work${ci}.d}[0], [y]               //C${ci}
	add y, y, 16
	st1 {work${ci}.d}[1], [y1]               //C${ci}
	add y1, y1, 16
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
