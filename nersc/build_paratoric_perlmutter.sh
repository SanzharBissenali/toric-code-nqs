#!/bin/bash
# Build ParaToric on a Perlmutter LOGIN node, importable from the tc-nqs conda env.
# Cluster twin of external/build_paratoric_local.sh (macOS). Everything lands in the
# gitignored external/ except one .pth in the env's site-packages. Idempotent.
#
# Verified 2026-08-05: gcc-15 toolchain via micromamba (PrgEnv gcc lacks C++23
# <print>; gcc-16 breaks std::println). -static-libstdc++ + RPATH make the .so
# self-contained — no LD_LIBRARY_PATH/preload needed at import time, in workers,
# or in submit wrappers. NATIVE_OPT is safe: login and compute nodes are the same
# AMD EPYC Milan generation.
set -euo pipefail

module load conda
conda activate tc-nqs
PY="$(which python)"
REPO="${REPO:-$HOME/toric-code-nqs}"
EXT="$REPO/external"
mkdir -p "$EXT" && cd "$EXT"

# 1. micromamba bootstrap + pinned gcc-15 toolchain
export MAMBA_ROOT_PREFIX="$EXT/mamba_root"; mkdir -p "$MAMBA_ROOT_PREFIX"
[ -x bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
[ -x "$EXT/tcqmc/bin/g++" ] || bin/micromamba create -y -r "$MAMBA_ROOT_PREFIX" -p "$EXT/tcqmc" \
  -c conda-forge "gcc=15" "gxx=15" "libboost-devel>=1.87" "hdf5>=1.14.3" cmake ninja

# 2. clone + upstream patch (lattice.cpp uses std::println without <print>)
[ -f ParaToric/CMakeLists.txt ] || git clone --recursive --depth 1 --shallow-submodules \
  https://github.com/palmbart/ParaToric
"$PY" - <<'EOF'
p = "ParaToric/src/lattice/lattice.cpp"; s = open(p).read()
if "#include <print>" not in s:
    open(p, "w").write(s.replace("#include <numeric>",
                                 "#include <numeric>\n#include <print>", 1))
EOF

# 3. configure + build (login-node polite -j8); FINDPYTHON=NEW is load-bearing —
#    pybind11 compatibility mode silently builds against the wrong python
rm -rf ParaToric/build ParaToric/python/paratoric/_paratoric*.so
export PATH="$EXT/tcqmc/bin:$PATH" CC="$EXT/tcqmc/bin/gcc" CXX="$EXT/tcqmc/bin/g++"
tcqmc/bin/cmake -S ParaToric -B ParaToric/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPARATORIC_ENABLE_NATIVE_OPT=ON \
  -DPARATORIC_LINK_MPI=OFF -DPARATORIC_BUILD_CLI=OFF -DPARATORIC_BUILD_TESTS=OFF \
  -DPARATORIC_BUILD_PYBIND=ON -DBOOST_ROOT="$EXT/tcqmc" -DHDF5_ROOT="$EXT/tcqmc" \
  -DPYBIND11_FINDPYTHON=NEW -DPython_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY" \
  -DCMAKE_MODULE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,-rpath,$EXT/tcqmc/lib -Wl,--disable-new-dtags" \
  -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,-rpath,$EXT/tcqmc/lib -Wl,--disable-new-dtags"
tcqmc/bin/cmake --build ParaToric/build -j8

# 4. importable via .pth (the only file outside external/)
echo "$EXT/ParaToric/python" > "$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/paratoric_local.pth"

# 5. honest import check — MUST touch the real API; __init__ swallows load
#    failures and __version__ reads "0+local" even on success
cd "$HOME" && "$PY" -c "from paratoric import extended_toric_code as etc; print('extension OK:', etc.get_sample)"
