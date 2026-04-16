# 3D Unstructured Finite Element Poisson Solve

Your task is to implement a 3D finite element poisson solve. No code
exists yet. Details are below. Start with building and installing the
latest stable trilinos configured specifically for this project after
we have finished planning.

## Scope

- 3D FE Poisson solver: `-div(kappa * grad u) = f`.
- Element support: `HEX8` (Q1) only for now. The assembly kernel
  should be templated on the element type so other shapes can be
  added later.
- Must support MPI. Every test must pass on `np=1` **and** `np=4`.
  All `mpirun`/`mpiexec` invocations run as the unprivileged `demo`
  user inside the container (UID 1000). OpenMPI's root-refusal safety
  check is therefore satisfied by construction; the build and test
  recipes rely on that default and do not pass
  `--allow-run-as-root`.
- Must be performance-portable via Kokkos (OpenMP backend required
  today; GPU-capable posture — the code must flip to CUDA/HIP with a
  Kokkos rebuild and nothing else).
- Avoid virtual functions on device — they complicate device-side
  dispatch.

## Mesh

- Use Trilinos `STK` for the mesh.
- Use the modern **unified field API** throughout.
- Turn aura ghosts on (`stk::mesh::BulkData::AUTO_AURA`) so FE
  assembly can iterate owned + aura elements.
- Build the cube mesh through `stk::io::StkMeshIoBroker` with
  `"generated:NxNxN|sideset:xXyYzZ"`. The generator partitions along
  Z and rejects `n < np`, so any test that runs at `np=4` needs
  `n ≥ 4`. MueLu additionally aborts with a KokkosBlas dimension
  mismatch when a rank hands it a zero-row matrix, which happens
  on small `n` once boundary-condensation leaves a **rank's** slab
  with no interior DOFs — pick `n ≥ 2·np` for any full-pipeline test to be
  safe (n=8 is what we use at np=4).

### STK unified field access — canonical usage

Strong index types (`stk::mesh::ComponentIdx`, `CopyIdx`, `ScalarIdx`)
are required — distinct types with *explicit* `int` constructors.
User-defined literals `_comp`, `_copy`, `_scalar` are provided for
readability.

Host access — both template params default to `ReadOnly`/`HostSpace`:

    const double x = coord_field.data().entity_values(node)(0_comp);
    solution_field.data<stk::mesh::ReadWrite>().entity_values(node)(0_comp) = 42.0;

### Device access — NgpMesh + FastMeshIndex

On device, prefer `stk::mesh::FastMeshIndex` (a
`{bucket_id, bucket_ord}` POD from `stk_mesh/base/Types.hpp`) over
`stk::mesh::Entity`. The device
`FieldData::entity_values(const FastMeshIndex&)` overload indexes
directly into bucket storage — no Entity→index lookup.

Node-loop pattern (e.g. zero a nodal field on owned nodes):

    auto& ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk);
    auto sol_data  = sol_field.data<stk::mesh::ReadWrite, stk::ngp::DeviceSpace>();

    stk::mesh::for_each_entity_run(
        ngp_mesh, stk::topology::NODE_RANK, meta.locally_owned_part(),
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& node_idx) {
          sol_data.entity_values(node_idx)(0_comp) = 0.0;
        });

Element-gather pattern — the canonical way to read nodal coords
inside the assembly kernel, straight from the live field into a
stack-allocated C array:

    auto& ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk);
    auto coords    = coord_field.data<stk::mesh::ReadOnly, stk::ngp::DeviceSpace>();

    stk::mesh::for_each_entity_run(
        ngp_mesh, stk::topology::ELEM_RANK,
        meta.locally_owned_part() | meta.aura_part(),
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& elem_idx) {
          double xyz[8][3];  // stack scratch — no Kokkos::View here

          const auto nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, elem_idx);
          for (int n = 0; n < 8; ++n) {
            const auto node_idx = ngp_mesh.fast_mesh_index(nodes[n]);
            const auto vals    = coords.entity_values(node_idx);
            xyz[n][0] = vals(0_comp);
            xyz[n][1] = vals(1_comp);
            xyz[n][2] = vals(2_comp);
          }
          // ... element-local compute + scatter ...
        });

**Sync is automatic — that is the whole point of the unified API.**
Every call to `field.data<AccessTag, Space>()` performs the correct
transfer for the requested access tag (`ReadOnly`, `ReadWrite`,
`OverwriteAll`) and memory space (`HostSpace`, `DeviceSpace`) before
returning the `FieldData`. Re-acquire the `FieldData` with a fresh
`data<...>()` call at each use site and the sync state takes care of
itself.

For the mesh itself, call `get_updated_ngp_mesh(bulk)` outside a
mesh modification cycle.

## Linear algebra & solver

- Pull in Tpetra, Belos, MueLu, Ifpack2, Amesos2.
- GMRES via Belos `PseudoBlockGmresSolMgr`.
- MueLu smoothed-aggregation multigrid preconditioner. Chebyshev
  smoother on every level. Coarse solve is **Two-stage
  Gauss-Seidel** (Ifpack2 relaxation type
  `"Two-stage Gauss-Seidel"`, 10 sweeps) — a hybrid relaxation
  whose inner triangular solve is approximated by Jacobi inner
  iterations, giving the same node-level parallelism as Chebyshev.
  Every level is therefore GPU-portable.
- Defaults (exposed on a solver `Options` struct): convergence tol
  `1e-10`, max iters `500`, restart (`Num Blocks`) `50`.
- MueLu configuration is an inline `Teuchos::ParameterList` in C++
  — no external XML driver.

## Shape functions

- Shape functions and their derivatives are **compile-time lookup
  tables** evaluated at the integration points.
- HEX8 quadrature is 2×2×2 Gauss–Legendre → `NQP = 8`. Other
  element traits pick their own rule, but any new one must still
  expose the trait protocol below.
- Verify via polynomial-reproducing tests.

## Assembly kernel

One device kernel with three explicit stages, executed via
`stk::mesh::for_each_entity_run` over `ELEM_RANK` with the
owned+aura selector:

1. **Gather** — read nodal coords from STK fields on device using
   the element-gather pattern above. All local scratch is
   stack-allocated on the thread: plain C arrays sized by the
   compile-time element trait. The gather reads straight from the
   live `coord_field` on `DeviceSpace` per the canonical pattern.
2. **Compute** — assemble the local element LHS (`Ke`) and RHS
   (`fe`) entirely on the thread's stack. 3×3 Jacobian closed-form
   inverse.
3. **Scatter** — `Kokkos::atomic_add` into
   `matrix.getLocalMatrixDevice()` values and
   `rhs.getLocalViewDevice(Tpetra::Access::ReadWrite)`. Two
   `Teuchos::OrdinalTraits<LocalOrdinal>::invalid()` cases:
   - `row_lids(e, i) == invalid()` — aura row not owned by us;
     skip (the owning rank picks it up).
   - `col_lids(e, j) == invalid()` — boundary column; lift into
     the RHS per §Dirichlet boundary conditions,
     `b_i -= Ke[i][j] * g(x_j)`, with `g(x_j)` read from the
     boundary_map-keyed prescribed field.
   Otherwise atomic-add `Ke[i][j]` into the matrix.

### Element trait protocol

Each element type is a trait struct exposing, as `static constexpr`:

    NDIM, NNODES, NQP
    N[NQP][NNODES]                    // shape values at QPs
    DN_DXI[NQP][NNODES][NDIM]         // reference-space derivatives
    QP_WEIGHTS[NQP]

## Dirichlet boundary conditions

Dirichlet is the only BC type in scope. No Neumann, Robin, or mixed
BCs — every face of every test mesh carries a prescribed `u` value
(possibly zero). `f` enters only as a volumetric source.

- Dirichlet BCs are **condensed out of the linear system**: the
  prescribed nodes are removed from the solve entirely, and the
  reduced system on interior DOFs is what goes to Belos/MueLu.
  **All four maps on the solve matrix (row, col, domain, range)
  hold interior DOFs only** — row/domain/range are owned interior,
  col is owned + ghosted interior. The owned-interior GID list for
  the row map comes from `meta.locally_owned_part() &
  !boundary_selector` at `NODE_RANK`; STK's
  `parallel_owner_rank` is not needed. Keep a *separate*
  `boundary_map` alongside for the boundary GIDs (owned + ghosted)
  that owned+aura elements reference; it exists purely to index
  the device-resident prescribed-value field. Putting boundary
  GIDs into the matrix's col_map looks tempting but breaks
  Tpetra: at np=1 `fillComplete` asserts on "remote col GIDs in a
  single-process communicator", and at np>1 MueLu segfaults
  inside the Galerkin R·A·P once coarsening kicks in (~n≥10).
- Find boundary nodes via the STK sidesets the mesh already
  carries. The `"generated:NxNxN|sideset:xXyYzZ"` source creates
  six side parts on the `x⁻, x⁺, y⁻, y⁺, z⁻, z⁺` faces; grab them
  from `MetaData` by name and union them into a single
  `stk::mesh::Selector`. Drive every boundary loop (DofMap
  exclusion, RHS lift, post-solve reconstruction) with that
  selector directly.
- Both the CrsGraph build and the assembly kernel iterate
  **owned + aura** elements (`locally_owned_part() | aura_part()`).
  Owned-only iteration silently drops stiffness entries from
  aura-side neighbours into the rank's owned rows on shared faces
  — np=1 still looks fine, np>1 returns a wrong solution.
- Because boundary DOFs are excluded from the row *and* col maps,
  the scatter kernel's existing `row_lids(e, i) == invalid()`
  check skips aura-owned rows and `col_lids(e, j) == invalid()`
  flags the boundary-column case. For each element column `j`
  whose col_lid is invalid, lift to the RHS:
  `b_i -= Ke[i][j] * g(x_j)` — `g(x_j)` comes from a
  device-resident nodal field keyed by **boundary_map LID**,
  populated once from the sideset selector at setup. For other
  columns, `atomic_add` `Ke[i][j]` as usual.
- After the solve, reconstruct the full nodal solution field:
  `parallel_for` on owned nodes, filling interior nodes from the
  solve result and writing `g(x)` into nodes that fall under the
  boundary selector.
- All of the above runs device-side via the NGP patterns from
  §STK unified field access and obeys the host-loop policy.

The reduced operator `A_II` is SPD and well-conditioned, which is
what the Belos/MueLu stack above expects.

## Host-loop policy

**Every hot-path loop runs on the device.** Any loop whose
iteration count scales with mesh DOFs/elements and runs more than
once (per solve, per time step, per post-processing pass) is
expressed as a `Kokkos::parallel_for` or `Kokkos::parallel_reduce`
on `Kokkos::DefaultExecutionSpace` so the code flips to GPU with a
Kokkos rebuild and nothing else.

This explicitly includes:
- the assembly kernel,
- Dirichlet BC application,
- error-norm integration (L² / H¹ reduction over elements),
- any user-facing post-processing (stress recovery, flux
  evaluation, node-wise function evaluation for visualization).

Host-serial loops are acceptable **only** for one-time,
mesh-setup-sized work STK or Tpetra require on host:
- constructing `DofMap` row/col GID lists,
- `CrsGraph::insertGlobalIndices` (Tpetra-host-only by contract).

## Verification & deliverable

**Patch tests first.** On `np=1` and `np≥2`: for every linear
`u(x,y,z) = a + b*x + c*y + d*z` prescribed as Dirichlet data with
`f ≡ 0`, the FE solution must reproduce `u` to `1e-10` L∞ nodal
error. Constant and linear fields must both pass.

**End deliverable — Q1 MMS convergence sweep.**

- Domain: cube `[0, L]^3` with `L = 1` (unit cube).
- Manufactured solution:
  `u(x, y, z) = sin(π x) * sin(π y) * sin(π z)`. Vanishes on every
  face.
- Source: `f = 3 * π^2 * u` (matches `-div grad u`).
- BCs: **Dirichlet on all six faces** (homogeneous, `u = 0`) — no
  Neumann or Robin segments anywhere in the problem.
- Resolutions: `n ∈ {10, 20, 40, 80}` (cube is `n × n × n` hex8).
  `n = 160` is out of scope — ≈4M DOFs/solve is too slow for a
  demo. A fifth rung behind a separate gtest label ("mms_full") is
  fine later.
- Error metrics: both
  - L²: `||u_h − u||_{L²}` = `sqrt(∫ (u_h − u)² dx)`,
  - H¹ seminorm: `|u_h − u|_{H¹}` = `sqrt(∫ |∇u_h − ∇u|² dx)`,
  integrated on the physical domain (unit cube). Use one
  element-wise `parallel_reduce` per norm, reusing the Jacobian
  inverse from assembly to map `DN_DXI` into physical gradients.
- Assertions on every consecutive pair of rungs:
  - L² rate `log2(e_L2_coarse / e_L2_fine)` lies in `[1.8, 2.2]`
    — Q1 gives `O(h²)`.
  - H¹ rate `log2(e_H1_coarse / e_H1_fine)` lies in `[0.8, 1.2]`
    — Q1 gives `O(h)`.
- Must pass on `np=1` and `np=4`.
- The test emits a CSV of `(n, l2_error, h1_error)` rows (header
  line `n,l2_error,h1_error`) to a path given by the env var
  `MMS_CONVERGENCE_CSV`; when unset the test skips writing. The
  conventional output path is
  `results/poisson_mms_q1_convergence.csv`.
- A log-log plot of both curves vs N with an `N⁻²` reference line
  is produced by `scripts/plot_mms_convergence.py <csv> [out.png]`.
  Conventional output:
  `results/poisson_mms_q1_convergence.png`. Run it after the test
  to regenerate the figure.
- **Gold comparison (final verification).** After the sweep passes,
  diff the emitted CSV against `gold/poisson_mms_q1_convergence.csv`
  (a checked-in reference from a known-good run):
  `diff -u gold/poisson_mms_q1_convergence.csv results/poisson_mms_q1_convergence.csv`.
  Expect each error norm to agree with gold to **two significant
  figures**; everything below that is reduction-order noise. Also
  eyeball `gold/poisson_mms_q1_convergence.png` against the freshly
  generated `results/poisson_mms_q1_convergence.png` — curves should
  overlap and the fitted slopes in the legend should match
  `L² = 1.99 / H¹ = 1.02` to two decimals.

## Test suite

Every test runs on `np=1` **and** `np=4` unless noted.

- `test_hex8_shape` — partition of unity, ∂N/∂ξ reproduction, QP
  weights sum to the reference-element volume.
- `test_mesh` — owned/aura counts, coord-field visibility,
  element→node gather map.
- `test_dofmap_graph` — row/col/boundary map sizes, every element
  node resolves in exactly one of col_map or boundary_map, the
  centre interior node of an `n=4` cube has the full 27-stencil
  (all 27 interior nodes couple through its 8 surrounding
  elements). (An n=2 mesh has one interior node with no interior
  neighbours under the condensed scheme, so the 27-stencil lives
  at n=4, not n=2.)
- `test_assembly` — `Ke` symmetry, zero row-sum on interior rows,
  analytic diagonal sum for a unit cube.
- `test_dirichlet` — condensation reproduces the prescribed
  nodal value after a solve.
- `test_patch` — linear solutions reproduced to `1e-10`.
- `test_solver` — `u ≡ 1` reproduced with `f ≡ 0` and `u = 1` BCs.
- `test_mms_convergence` — the Q1 sweep above.
- `test_smoke` — compile-and-link smoke.

## Process

- Work incrementally in small, meaningful commits. 
- Think through how you want to plan out tasks, structuring them into meaningful standlone efforts. 
- Involve the user in this planning process and clear up any uncertainties you may have regarding the prompt.md
- After each task verify the accompanying unit tests pass if applicable and commit.
- Make sure to use clang lint build prior to committing and
  address all errors.
- After every commit, regenerate `CLAUDE.md` at the repo root so it
  reflects the new state of the tree: build + test commands, the
  current layout under `src/` and `tests/`, the Trilinos install
  path, and anything else a fresh session needs to be productive.
  Create the file on the first commit; overwrite it on every
  subsequent commit. Include it in the same commit as the change.
- Ask the user if they want to review each phase or if you should just work autonomously till the end.
- Waiting for long-running jobs (the Trilinos build is ~30 min):
  prefer `Bash(run_in_background=true)` over `Monitor` with a
  log-pattern regex. The Bash tool returns the process exit code
  on completion automatically — no "is it done yet?" grep needed.
  Reach for `Monitor` only when you want a stream of named events
  (every FAIL in a long test loop), not a single terminal signal.

## Build

- Compilers are `mpicc` / `mpicxx` (OpenMPI wrappers). They wrap
  clang via `OMPI_CC=clang` and `OMPI_CXX=clang++`, set in the
  preset environment and the Dockerfile — nothing else invokes
  the compiler directly.
- CMake + Ninja.
- Adhere to `.clang-format` / `.clang-tidy` in the repo.
- Update `.clang-tidy` to prefer our headers listed first and add
  a header filter regex only for our files.
- Prefer CMake presets. Four: `dev` (Debug), `dev-lint` (Debug +
  `ENABLE_CLANG_TIDY=ON`, separate `build-lint/` dir), `release`
  (RelWithDebInfo — use for MMS timing and the gold-diff pass),
  and `asan` (Debug + `-fsanitize=address,undefined`). All set
  `BUILD_TESTING=ON`.
- Install Trilinos 17.0.0 + gtest 1.14.0 via
  `scripts/build_trilinos.sh` — a shallow clone into `tpls/trilinos`,
  install under `tpls/install/{trilinos,gtest}`. The script is the
  source of truth for the package set (Tpetra, Belos, MueLu, Ifpack2,
  Amesos2, Zoltan2, STK{Mesh,IO,Topology,Util,Search}, SEACASIoss)
  and the flags required to get a usable install on this stack:
  C++20, `Tpetra_INST_INT_LONG_LONG=ON` / `Tpetra_INST_INT_INT=OFF`
  (to line up with `stk::mesh::EntityId`), `Kokkos_ENABLE_OPENMP=ON`,
  `Kokkos_ARCH_NATIVE=OFF`, `TPL_ENABLE_BoostLib=OFF` (header-only
  boost is enough for STK), `TPL_Netcdf_PARALLEL=OFF`, and static
  libs with PIC. A stamp file under the install prefix makes
  re-runs a no-op.

### clang-tidy through the `mpicxx` wrapper

CMake caches `mpicxx`'s implicit `-I` paths in
`CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES` and elides them from
`compile_commands.json`, so clang-tidy (which bypasses the wrapper)
can't find `mpi.h`. Fix in the root `CMakeLists.txt`: parse
`mpicxx -showme:compile`, `list(REMOVE_ITEM ...)` from the implicit
list, then re-attach via `target_include_directories(... SYSTEM
INTERFACE ...)`. Nothing else works — `MPI::MPI_CXX` properties
get stripped the same way.
