#!/usr/bin/env bash
#
# scripts/build_trilinos.sh
#
# Shallow-clone and install, under tpls/install/:
#   1. googletest v1.14.0 (STK 17 is incompatible with gtest >=1.15)
#   2. Trilinos trilinos-release-17-0-0, restricted to the packages we need:
#      Teuchos, Kokkos, KokkosKernels, Tpetra, Belos, Ifpack2, MueLu,
#      Amesos2, Zoltan2, STK{Mesh,IO,Topology,Util,Search}, SEACASIoss.
#
# Idempotent: cmake + ninja short-circuit on an already-built tree,
# so re-running is safe (and takes ~7 seconds once the install exists).
# Compilers come from the MPI wrappers (mpicc / mpicxx), which wrap clang /
# clang++ via OMPI_CC / OMPI_CXX set in the container environment.
#
# Override JOBS=N to change the parallelism; defaults to $(nproc).
#
# Pass --clean to wipe the install prefixes and build dirs (but keep
# the shallow clones) before rebuilding. Use this to verify a fresh
# build still works without paying the git-clone cost again.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

TPLS="${ROOT}/tpls"
BUILD_DIR="${TPLS}/build"
INSTALL_DIR="${TPLS}/install"
JOBS="${JOBS:-$(nproc)}"

CLEAN=0
for arg in "$@"; do
  case "${arg}" in
    --clean) CLEAN=1 ;;
    *) printf 'unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

GTEST_TAG="v1.14.0"
GTEST_SRC="${TPLS}/googletest"
GTEST_INSTALL="${INSTALL_DIR}/gtest"

TRILINOS_TAG="trilinos-release-17-0-0"
TRILINOS_SRC="${TPLS}/trilinos"
TRILINOS_INSTALL="${INSTALL_DIR}/trilinos"

if [[ "${CLEAN}" -eq 1 ]]; then
  printf '[build_trilinos] cleaning install + build dirs (keeping source clones)\n'
  rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
fi

mkdir -p "${TPLS}" "${BUILD_DIR}" "${INSTALL_DIR}"

log() {
  # ANSI escape codes only when stdout is a TTY, so anyone tailing the
  # log file (or a Monitor regex) sees plain "[build_trilinos] ..." at
  # the start of the line.
  if [[ -t 1 ]]; then
    printf '\033[1;34m[build_trilinos]\033[0m %s\n' "$*"
  else
    printf '[build_trilinos] %s\n' "$*"
  fi
}

# ---------------------------------------------------------------------- gtest
if [[ ! -d "${GTEST_SRC}/.git" ]]; then
  log "shallow-cloning googletest ${GTEST_TAG}"
  git clone --depth 1 --branch "${GTEST_TAG}" \
    https://github.com/google/googletest.git "${GTEST_SRC}"
fi

log "configuring gtest"
cmake -S "${GTEST_SRC}" -B "${BUILD_DIR}/gtest" -G Ninja \
  -DCMAKE_INSTALL_PREFIX="${GTEST_INSTALL}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_GMOCK=OFF \
  -DINSTALL_GTEST=ON

log "building + installing gtest"
cmake --build "${BUILD_DIR}/gtest" -j "${JOBS}"
cmake --install "${BUILD_DIR}/gtest"

# -------------------------------------------------------------------- trilinos
if [[ ! -d "${TRILINOS_SRC}/.git" ]]; then
  log "shallow-cloning trilinos ${TRILINOS_TAG}"
  git clone --depth 1 --branch "${TRILINOS_TAG}" \
    https://github.com/trilinos/Trilinos.git "${TRILINOS_SRC}"
fi

log "configuring trilinos (this takes a few minutes)"
cmake -S "${TRILINOS_SRC}" -B "${BUILD_DIR}/trilinos" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${TRILINOS_INSTALL}" \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_PREFIX_PATH="${GTEST_INSTALL}" \
  -DGTest_ROOT="${GTEST_INSTALL}" \
  \
  -DTrilinos_ENABLE_TESTS=OFF \
  -DTrilinos_ENABLE_EXAMPLES=OFF \
  -DTrilinos_ENABLE_ALL_PACKAGES=OFF \
  -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
  -DTrilinos_ENABLE_SECONDARY_TESTED_CODE=OFF \
  -DTrilinos_ASSERT_MISSING_PACKAGES=OFF \
  \
  -DTrilinos_ENABLE_Teuchos=ON \
  -DTrilinos_ENABLE_Kokkos=ON \
  -DTrilinos_ENABLE_KokkosKernels=ON \
  -DTrilinos_ENABLE_Tpetra=ON \
  -DTrilinos_ENABLE_Belos=ON \
  -DTrilinos_ENABLE_Ifpack2=ON \
  -DTrilinos_ENABLE_MueLu=ON \
  -DTrilinos_ENABLE_Amesos2=ON \
  -DTrilinos_ENABLE_Zoltan2=ON \
  -DTrilinos_ENABLE_STKMesh=ON \
  -DTrilinos_ENABLE_STKIO=ON \
  -DTrilinos_ENABLE_STKTopology=ON \
  -DTrilinos_ENABLE_STKUtil=ON \
  -DTrilinos_ENABLE_STKSearch=ON \
  -DTrilinos_ENABLE_SEACASIoss=ON \
  \
  -DTrilinos_ENABLE_OpenMP=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_NATIVE=OFF \
  \
  -DTpetra_INST_INT_LONG_LONG=ON \
  -DTpetra_INST_INT_INT=OFF \
  -DTpetra_INST_DOUBLE=ON \
  \
  -DTPL_ENABLE_MPI=ON \
  -DTPL_ENABLE_Boost=ON \
  -DTPL_ENABLE_BoostLib=OFF \
  -DTPL_ENABLE_HDF5=ON \
  -DTPL_ENABLE_Netcdf=ON \
  -DTPL_Netcdf_PARALLEL=OFF \
  -DTPL_ENABLE_BLAS=ON \
  -DTPL_ENABLE_LAPACK=ON \
  -DTPL_ENABLE_ParMETIS=ON \
  -DTPL_ENABLE_METIS=ON \
  -DTPL_ENABLE_Pthread=OFF

log "building trilinos with ${JOBS} jobs"
cmake --build "${BUILD_DIR}/trilinos" -j "${JOBS}"

log "installing trilinos"
cmake --install "${BUILD_DIR}/trilinos"

log "done. Install prefix: ${TRILINOS_INSTALL}"
