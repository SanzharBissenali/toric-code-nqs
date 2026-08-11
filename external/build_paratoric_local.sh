#!/bin/bash
# Build ParaToric + its python bindings on macOS with Homebrew LLVM.
#
# Toolchain choice: brew's Boost/HDF5 are libc++ builds, so the compiler must be
# brew clang (libc++), NOT gcc (libstdc++) -- mixing the two C++ runtimes fails
# at link/run time. ParaToric is tested upstream with Clang 20.
# Prereq (user-run): brew install llvm boost hdf5 ninja
set -euo pipefail
cd "$(dirname "$0")"

LLVM=$(brew --prefix llvm)
VENV_PY=$(cd .. && pwd)/.venv/bin/python

# upstream bug (reported): lattice.cpp calls std::println without #include <print>
python3 - <<'PY'
p = "ParaToric/src/lattice/lattice.cpp"
s = open(p).read()
if "#include <print>" not in s:
    s = s.replace("#include <numeric>", "#include <numeric>\n#include <print>", 1)
    open(p, "w").write(s)
PY

# our membrane observable (fredenhagen_marcu_membrane) — committed patch, idempotent
if git -C ParaToric apply --reverse --check ../paratoric_membrane.patch 2>/dev/null; then
  echo "[build] paratoric_membrane.patch already applied"
elif git -C ParaToric apply ../paratoric_membrane.patch 2>/dev/null; then
  echo "[build] applied paratoric_membrane.patch"
else
  echo "[build] ERROR: paratoric_membrane.patch failed to apply"; exit 1
fi

# tau-warning via C stdio (segfault fix, see patch header) — idempotent
if git -C ParaToric apply --reverse --check ../paratoric_stdio_taulog.patch 2>/dev/null; then
  echo "[build] paratoric_stdio_taulog.patch already applied"
elif git -C ParaToric apply ../paratoric_stdio_taulog.patch 2>/dev/null; then
  echo "[build] applied paratoric_stdio_taulog.patch"
else
  echo "[build] ERROR: paratoric_stdio_taulog.patch failed to apply"; exit 1
fi

export CC="$LLVM/bin/clang" CXX="$LLVM/bin/clang++"
# brew llvm ships its own libc++; point the linker at it and bake the rpath
export LDFLAGS="-L$LLVM/lib/c++ -Wl,-rpath,$LLVM/lib/c++"

rm -rf ParaToric/build ParaToric/python/paratoric/_paratoric*.so
cmake -S ParaToric -B ParaToric/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPARATORIC_ENABLE_NATIVE_OPT=ON \
  -DPARATORIC_LINK_MPI=OFF -DPARATORIC_BUILD_CLI=OFF -DPARATORIC_BUILD_TESTS=ON \
  -DPARATORIC_BUILD_PYBIND=ON \
  -DBOOST_ROOT="$(brew --prefix boost)" -DHDF5_ROOT="$(brew --prefix hdf5)" \
  -DPYBIND11_FINDPYTHON=NEW -DPython_EXECUTABLE="$VENV_PY" -DPython3_EXECUTABLE="$VENV_PY"
cmake --build ParaToric/build -j"$(sysctl -n hw.ncpu)"

ls -l ParaToric/python/paratoric/_paratoric*.so
echo "--- import check (venv python) ---"
PYTHONPATH=ParaToric/python "$VENV_PY" -c \
  "import paratoric; print('get_sample:', paratoric.extended_toric_code.get_sample)"   # raises if the .so failed
