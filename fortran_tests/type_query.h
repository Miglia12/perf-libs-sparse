! SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
! affiliates <open-source-office@arm.com></text>
!
! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

    ! Find out what type we're dealing with (it seems impossible to use the
    ! preprocessor to do '#if FTYPE==real', for example, so use storage_size instead!
    integer :: typeid
    if (rk .eq. kind(1.0) .and. storage_size(dummy) .eq. 32) then
        ! S
        typeid = 1
    else if (rk .eq. kind(1.d0) .and. storage_size(dummy) .eq. 64) then
        ! D
        typeid = 2
    else if (rk .eq. kind(1.0) .and. storage_size(dummy) .eq. 64) then
        ! C
        typeid = 3
    else
        ! Z
        typeid = 4
    end if
