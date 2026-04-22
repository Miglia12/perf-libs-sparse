# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
# affiliates <open-source-office@arm.com></text>
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# cmake/blas_fallback.cmake
#
# Provides:
#   - function: perflibs_setup_blas(...)
#
# Behavior:
#   - Creates package-scoped BLAS/CBLAS interface targets for the current ABI.
#   - Uses ArmPL when requested.
#   - Otherwise requires a system OpenBLAS (via CMake config, pkg-config, or find_library).
#   - If SPARSE_ENABLE_INT64=ON, fails unless it finds an ILP64-looking OpenBLAS.

include_guard(GLOBAL)
include(CMakeParseArguments)

function(_perflibs_ensure_interface_target target_name)
  if(NOT TARGET "${target_name}")
    add_library("${target_name}" INTERFACE IMPORTED GLOBAL)
  endif()
endfunction()

function(_perflibs_use_sparse_armpl package_name blas_target cblas_target armpl_lib_basename)
  if(NOT DEFINED ENV{ARMPL_DIR})
    message(FATAL_ERROR "SPARSE_USE_ARMPL=ON but environment variable ARMPL_DIR is not set.")
  endif()

  set(_armpl_root "$ENV{ARMPL_DIR}")
  set(_inc "${_armpl_root}/include")
  set(_libdir "${_armpl_root}/lib")

  if(NOT EXISTS "${_inc}/cblas.h")
    message(FATAL_ERROR "SPARSE_USE_ARMPL=ON but '${_inc}/cblas.h' not found. Check ARMPL_DIR='${_armpl_root}'.")
  endif()

  if(WIN32)
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded")
      set(_armpl_filename "lib${armpl_lib_basename}.lib")
    else()
      set(_armpl_filename "${armpl_lib_basename}.lib")
    endif()

    set(_armpl_lib "${_libdir}/${_armpl_filename}")
    if(NOT EXISTS "${_armpl_lib}")
      message(FATAL_ERROR
        "SPARSE_USE_ARMPL=ON but expected ArmPL library not found: ${_armpl_lib}\n"
        "Check ARMPL_DIR and ARMPL_LIB_BASENAME."
      )
    endif()
  else()
    find_library(_armpl_lib
      NAMES "${armpl_lib_basename}" "lib${armpl_lib_basename}.so" "lib${armpl_lib_basename}.a"
      HINTS "${_libdir}"
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT _armpl_lib)
    message(FATAL_ERROR
      "SPARSE_USE_ARMPL=ON but could not find lib${armpl_lib_basename} in ${_libdir}. "
      "Set -DARMPL_LIB_BASENAME=... or check your ARMPL_DIR."
    )
  endif()

  set(_armpl_impl_target "${package_name}::armpl")
  if(NOT TARGET "${_armpl_impl_target}")
    add_library("${_armpl_impl_target}" UNKNOWN IMPORTED GLOBAL)
    set_target_properties("${_armpl_impl_target}" PROPERTIES
      IMPORTED_LOCATION "${_armpl_lib}"
      INTERFACE_INCLUDE_DIRECTORIES "${_inc}"
    )
  endif()

  if(WIN32)
    target_link_directories("${_armpl_impl_target}" INTERFACE "${_libdir}")
  endif()

  target_include_directories("${cblas_target}" INTERFACE "${_inc}")
  set_target_properties("${blas_target}" PROPERTIES INTERFACE_LINK_LIBRARIES "${_armpl_impl_target}")
  set_target_properties("${cblas_target}" PROPERTIES INTERFACE_LINK_LIBRARIES "${_armpl_impl_target}")

  set(PERFLIBS_OPENBLAS_DIR_HINT "" PARENT_SCOPE)
  set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "" PARENT_SCOPE)
  set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "" PARENT_SCOPE)
endfunction()

function(_perflibs_use_resolved_openblas package_name blas_target cblas_target openblas_libraries openblas_include_dirs)
  set(_openblas_impl_target "${package_name}::openblas")
  if(NOT TARGET "${_openblas_impl_target}")
    add_library("${_openblas_impl_target}" INTERFACE IMPORTED GLOBAL)
    set_target_properties("${_openblas_impl_target}" PROPERTIES
      INTERFACE_LINK_LIBRARIES "${openblas_libraries}"
      INTERFACE_INCLUDE_DIRECTORIES "${openblas_include_dirs}"
    )
  endif()

  target_include_directories("${cblas_target}" INTERFACE "${openblas_include_dirs}")
  set_target_properties("${blas_target}" PROPERTIES INTERFACE_LINK_LIBRARIES "${_openblas_impl_target}")
  set_target_properties("${cblas_target}" PROPERTIES INTERFACE_LINK_LIBRARIES "${_openblas_impl_target}")
endfunction()

function(_perflibs_fail_ilp64 msg)
  message(FATAL_ERROR
    "SPARSE_ENABLE_INT64=ON requires an ILP64 OpenBLAS (64-bit BLAS integers).\n"
    "${msg}\n"
    "Install an ILP64 OpenBLAS package (often provides 'openblas64_' and headers under include/openblas64),\n"
    "or configure with SPARSE_USE_ARMPL=ON and set ARMPL_DIR to an ArmPL ILP64 install.\n"
  )
endfunction()

function(_perflibs_use_system_openblas_strict
    package_name
    blas_target
    cblas_target
    want_ilp64
    openblas_dir_hint
    openblas_libraries_hint
    openblas_include_dirs_hint)
  if(openblas_libraries_hint AND openblas_include_dirs_hint)
    _perflibs_use_resolved_openblas(
      "${package_name}"
      "${blas_target}"
      "${cblas_target}"
      "${openblas_libraries_hint}"
      "${openblas_include_dirs_hint}"
    )
    return()
  endif()

  if(openblas_dir_hint)
    set(OpenBLAS_DIR "${openblas_dir_hint}")
  endif()

  find_package(OpenBLAS CONFIG QUIET)
  if(TARGET OpenBLAS::OpenBLAS OR OpenBLAS_FOUND)
    if(want_ilp64)
      _perflibs_fail_ilp64("Found an OpenBLAS CMake package, but it is usually LP64.")
    endif()

    if(TARGET OpenBLAS::OpenBLAS)
      get_target_property(_openblas_include_dirs OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
      if(_openblas_include_dirs AND NOT _openblas_include_dirs STREQUAL "_openblas_include_dirs-NOTFOUND")
        target_include_directories("${cblas_target}" INTERFACE "${_openblas_include_dirs}")
      endif()
      set_target_properties("${blas_target}" PROPERTIES INTERFACE_LINK_LIBRARIES OpenBLAS::OpenBLAS)
      set_target_properties("${cblas_target}" PROPERTIES INTERFACE_LINK_LIBRARIES OpenBLAS::OpenBLAS)
      if(DEFINED OpenBLAS_DIR)
        set(PERFLIBS_OPENBLAS_DIR_HINT "${OpenBLAS_DIR}" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "" PARENT_SCOPE)
      endif()
      return()
    endif()

    if(DEFINED OpenBLAS_LIBRARIES AND DEFINED OpenBLAS_INCLUDE_DIRS)
      _perflibs_use_resolved_openblas(
        "${package_name}"
        "${blas_target}"
        "${cblas_target}"
        "${OpenBLAS_LIBRARIES}"
        "${OpenBLAS_INCLUDE_DIRS}"
      )
      if(DEFINED OpenBLAS_DIR)
        set(PERFLIBS_OPENBLAS_DIR_HINT "${OpenBLAS_DIR}" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "" PARENT_SCOPE)
      else()
        set(PERFLIBS_OPENBLAS_DIR_HINT "" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "${OpenBLAS_LIBRARIES}" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "${OpenBLAS_INCLUDE_DIRS}" PARENT_SCOPE)
      endif()
      return()
    endif()

    message(FATAL_ERROR
      "Found an OpenBLAS CMake package, but it did not provide either "
      "OpenBLAS::OpenBLAS or usable OpenBLAS_LIBRARIES/OpenBLAS_INCLUDE_DIRS."
    )
  endif()

  include(FindPkgConfig)
  if(PKG_CONFIG_FOUND)
    if(want_ilp64)
      pkg_check_modules(PC_OPENBLAS64U QUIET openblas64_)
      if(PC_OPENBLAS64U_FOUND)
        _perflibs_use_resolved_openblas(
          "${package_name}"
          "${blas_target}"
          "${cblas_target}"
          "${PC_OPENBLAS64U_LINK_LIBRARIES}"
          "${PC_OPENBLAS64U_INCLUDE_DIRS}"
        )
        set(PERFLIBS_OPENBLAS_DIR_HINT "" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "${PC_OPENBLAS64U_LINK_LIBRARIES}" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "${PC_OPENBLAS64U_INCLUDE_DIRS}" PARENT_SCOPE)
        return()
      endif()

      pkg_check_modules(PC_OPENBLAS64 QUIET openblas64)
      if(PC_OPENBLAS64_FOUND)
        _perflibs_use_resolved_openblas(
          "${package_name}"
          "${blas_target}"
          "${cblas_target}"
          "${PC_OPENBLAS64_LINK_LIBRARIES}"
          "${PC_OPENBLAS64_INCLUDE_DIRS}"
        )
        set(PERFLIBS_OPENBLAS_DIR_HINT "" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "${PC_OPENBLAS64_LINK_LIBRARIES}" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "${PC_OPENBLAS64_INCLUDE_DIRS}" PARENT_SCOPE)
        return()
      endif()

      pkg_check_modules(PC_OPENBLAS QUIET openblas)
      if(PC_OPENBLAS_FOUND)
        _perflibs_fail_ilp64("Found pkg-config 'openblas' only (likely LP64); no ILP64 pkg-config entry (openblas64_/openblas64).")
      endif()
    else()
      pkg_check_modules(PC_OPENBLAS QUIET openblas)
      if(PC_OPENBLAS_FOUND)
        set(_openblas_include_dirs "${PC_OPENBLAS_INCLUDE_DIRS}")
        foreach(_d IN LISTS PC_OPENBLAS_INCLUDE_DIRS)
          if(EXISTS "${_d}/openblas/cblas.h")
            list(APPEND _openblas_include_dirs "${_d}/openblas")
          endif()
        endforeach()

        _perflibs_use_resolved_openblas(
          "${package_name}"
          "${blas_target}"
          "${cblas_target}"
          "${PC_OPENBLAS_LINK_LIBRARIES}"
          "${_openblas_include_dirs}"
        )
        set(PERFLIBS_OPENBLAS_DIR_HINT "" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "${PC_OPENBLAS_LINK_LIBRARIES}" PARENT_SCOPE)
        set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "${_openblas_include_dirs}" PARENT_SCOPE)
        return()
      endif()
    endif()
  endif()

  set(_openblas_names "")
  if(want_ilp64)
    list(APPEND _openblas_names openblas64_ openblas_ilp64 openblas64)
  else()
    list(APPEND _openblas_names openblas)
  endif()

  find_library(_openblas_lib NAMES ${_openblas_names})
  find_path(_openblas_inc NAMES cblas.h PATH_SUFFIXES
    include
    include/openblas
    include/openblas64
    openblas
    openblas64
  )

  if(want_ilp64)
    if(NOT _openblas_lib)
      _perflibs_fail_ilp64("Could not find an ILP64 OpenBLAS library (tried: openblas64_, openblas64, openblas_ilp64).")
    endif()
    if(NOT _openblas_inc)
      _perflibs_fail_ilp64("Could not find OpenBLAS headers (cblas.h). Looked under include/openblas64 etc.")
    endif()
  else()
    if(NOT _openblas_lib OR NOT _openblas_inc)
      message(FATAL_ERROR
        "Could not find a system OpenBLAS (BLAS/CBLAS) installation.\n"
        "Install it or configure with SPARSE_USE_ARMPL=ON and set ARMPL_DIR.\n"
      )
    endif()
  endif()

  _perflibs_use_resolved_openblas(
    "${package_name}"
    "${blas_target}"
    "${cblas_target}"
    "${_openblas_lib}"
    "${_openblas_inc}"
  )
  set(PERFLIBS_OPENBLAS_DIR_HINT "" PARENT_SCOPE)
  set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "${_openblas_lib}" PARENT_SCOPE)
  set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "${_openblas_inc}" PARENT_SCOPE)
endfunction()

function(perflibs_setup_blas)
  set(one_value_args
    PACKAGE_NAME
    BLAS_TARGET
    CBLAS_TARGET
    SPARSE_USE_ARMPL
    SPARSE_ENABLE_INT64
    ARMPL_LIB_BASENAME
    OPENBLAS_DIR_HINT
  )
  set(multi_value_args
    OPENBLAS_LIBRARIES_HINT
    OPENBLAS_INCLUDE_DIRS_HINT
  )
  cmake_parse_arguments(PERFLIBS "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  foreach(_required PACKAGE_NAME BLAS_TARGET CBLAS_TARGET SPARSE_USE_ARMPL SPARSE_ENABLE_INT64 ARMPL_LIB_BASENAME)
    if(NOT DEFINED PERFLIBS_${_required} OR PERFLIBS_${_required} STREQUAL "")
      message(FATAL_ERROR "perflibs_setup_blas: missing required argument ${_required}")
    endif()
  endforeach()

  _perflibs_ensure_interface_target("${PERFLIBS_BLAS_TARGET}")
  _perflibs_ensure_interface_target("${PERFLIBS_CBLAS_TARGET}")

  if(PERFLIBS_SPARSE_USE_ARMPL)
    message(STATUS "Using ArmPL (SPARSE_USE_ARMPL=ON)")
    _perflibs_use_sparse_armpl(
      "${PERFLIBS_PACKAGE_NAME}"
      "${PERFLIBS_BLAS_TARGET}"
      "${PERFLIBS_CBLAS_TARGET}"
      "${PERFLIBS_ARMPL_LIB_BASENAME}"
    )
  else()
    message(STATUS "Using system OpenBLAS")
    _perflibs_use_system_openblas_strict(
      "${PERFLIBS_PACKAGE_NAME}"
      "${PERFLIBS_BLAS_TARGET}"
      "${PERFLIBS_CBLAS_TARGET}"
      "${PERFLIBS_SPARSE_ENABLE_INT64}"
      "${PERFLIBS_OPENBLAS_DIR_HINT}"
      "${PERFLIBS_OPENBLAS_LIBRARIES_HINT}"
      "${PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT}"
    )
  endif()

  set(PERFLIBS_OPENBLAS_DIR_HINT "${PERFLIBS_OPENBLAS_DIR_HINT}" PARENT_SCOPE)
  set(PERFLIBS_OPENBLAS_LIBRARIES_HINT "${PERFLIBS_OPENBLAS_LIBRARIES_HINT}" PARENT_SCOPE)
  set(PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT "${PERFLIBS_OPENBLAS_INCLUDE_DIRS_HINT}" PARENT_SCOPE)
endfunction()
