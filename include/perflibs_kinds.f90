! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

module perflibs_kinds

integer, parameter :: perflibs_i4 = selected_int_kind(8)
integer, parameter :: perflibs_i8 = selected_int_kind(10)
integer, parameter :: perflibs_r32 = selected_real_kind(6, 37)
integer, parameter :: perflibs_r64 = selected_real_kind(15, 307)

end module perflibs_kinds
