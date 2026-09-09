"""
Resume-config guard (branch p3d/close-medium) -- config-dict level, no netket.

  1. check_resume_config (tc3d.io): the shared guard used by both tc3d.sweep
     and tc3d.train --resume.
       (a) legacy checkpoint (compute_dtype/inv_impl/qgt_solver absent, as
           written before these keys existed) + empty knobs now -> ok.
       (b) legacy checkpoint + an explicit float32 request now -> mismatch,
           exit(2).
       (c) explicit float32/dense/kernel checkpoint + the SAME explicit
           request -> ok.
       (d) same mismatch as (b), but allow_mismatch=True -> no exit.
       (e) a genuine non-lever mismatch (hx) always trips regardless of the
           lever normalization.
  2. apply_late_lever_defaults (tc3d.config): the Python-side mirror of the
     wrapper's TC3D_DEFAULTS_FILE block.
       (a) unset knobs + a defaults file -> filled from it.
       (b) RESUB_COUNT=1 (a requeue) -> untouched even with a matching file.
       (c) an explicit value already in cfg -> never overwritten by the file.
       (d) no file at TC3D_DEFAULTS_FILE -> untouched (e.g. /dev/null).

Run:  cd tests && ../.venv/bin/python test_resume_guard.py
"""
import json
import os
import tempfile

from tc3d.io import check_resume_config
from tc3d.config import apply_late_lever_defaults


def _write_curve(path, config):
    with open(path, "w") as f:
        json.dump({"completed_steps": 20, "name": "pt", "config": config, "curve": {}}, f)


_BASE = dict(dt=0.02, lr_min=0.002, n_iter=200, diag_shift=1e-3, hx=0.2, hy=0.0, hz=0.26, L=4)


def _expect_ok(curve_path, name, cfg, **kw):
    check_resume_config(curve_path, name, cfg, **kw)   # must not raise/exit


def _expect_exit2(curve_path, name, cfg, **kw):
    try:
        check_resume_config(curve_path, name, cfg, **kw)
    except SystemExit as e:
        assert e.code == 2, f"expected exit(2), got {e.code!r}"
        return
    raise AssertionError("expected SystemExit(2), guard did not fire")


# 1 ---------------------------------------------------------------------------
def test_check_resume_config():
    with tempfile.TemporaryDirectory() as d:
        curve_path = os.path.join(d, "pt.curve.json")

        # (a) legacy checkpoint (no compute_dtype/inv_impl/qgt_solver keys at all)
        # + empty knobs now -> the missing-vs-None normalization must agree
        legacy = dict(_BASE)                      # no lever keys, as pre-lever code wrote it
        _write_curve(curve_path, legacy)
        now_empty = dict(_BASE, compute_dtype=None, inv_impl="conv", qgt_solver=None)
        _expect_ok(curve_path, "pt", now_empty)
        print("[1a] legacy checkpoint + empty knobs: ok (no false mismatch)")

        # (b) legacy checkpoint + an explicit float32 request -> real mismatch
        now_f32 = dict(_BASE, compute_dtype="float32", inv_impl="dense", qgt_solver="cholesky")
        _expect_exit2(curve_path, "pt", now_f32)
        print("[1b] legacy checkpoint + compute_dtype=float32 now: exit(2) as expected")

        # (c) explicit f32/dense/kernel checkpoint + the SAME request -> ok
        explicit = dict(_BASE, compute_dtype="float32", inv_impl="dense", qgt_solver="kernel")
        _write_curve(curve_path, explicit)
        _expect_ok(curve_path, "pt", dict(explicit))
        print("[1c] explicit float32/dense/kernel checkpoint + identical request: ok")

        # (c') the SAME checkpoint against a DIFFERENT explicit solver -> mismatch
        _expect_exit2(curve_path, "pt", dict(explicit, qgt_solver="cholesky"))
        print("[1c'] explicit checkpoint + a genuinely different qgt_solver: exit(2)")

        # (d) (b)'s mismatch, but allow_mismatch=True -> no exit
        _write_curve(curve_path, legacy)
        _expect_ok(curve_path, "pt", now_f32, allow_mismatch=True)
        print("[1d] same mismatch as (b) with allow_mismatch=True: no exit")

        # (e) a genuine non-lever mismatch (hx) trips regardless of lever normalization
        _expect_exit2(curve_path, "pt", dict(now_empty, hx=0.4))
        print("[1e] hx mismatch still trips independent of the lever keys")

        # no checkpoint on disk yet -> always a no-op, whatever cfg says
        os.remove(curve_path)
        _expect_ok(curve_path, "pt", now_f32)
        print("[1] no checkpoint on disk: no-op (fresh point/run)")


# 2 ---------------------------------------------------------------------------
def test_late_lever_defaults():
    saved = {k: os.environ.get(k) for k in ("RESUB_COUNT", "TC3D_DEFAULTS_FILE", "PSCRATCH")}
    try:
        with tempfile.TemporaryDirectory() as d:
            defaults_file = os.path.join(d, "defaults.env")
            with open(defaults_file, "w") as f:
                f.write("# comment\nINV_IMPL=dense\nCOMPUTE_DTYPE=float32\nQGT_SOLVER=\n")
            os.environ["TC3D_DEFAULTS_FILE"] = defaults_file

            # (a) unset knobs + first start -> filled from the file (QGT_SOLVER
            # is blank in the file, same as the wrapper's `[ -n ... ]` check --
            # left untouched)
            os.environ.pop("RESUB_COUNT", None)
            cfg = {"L": 4, "hx": 0.2}
            apply_late_lever_defaults(cfg, prefix="test")
            assert cfg.get("compute_dtype") == "float32", cfg
            assert cfg.get("inv_impl") == "dense", cfg
            assert cfg.get("qgt_solver") is None, cfg
            print("[2a] first start + defaults file: compute_dtype/inv_impl filled, "
                  "blank qgt_solver left unset")

            # (b) RESUB_COUNT=1 (a requeue) -> untouched even though the file matches
            os.environ["RESUB_COUNT"] = "1"
            cfg2 = {"L": 4, "hx": 0.2}
            apply_late_lever_defaults(cfg2, prefix="test")
            assert cfg2.get("compute_dtype") is None and cfg2.get("inv_impl") is None
            print("[2b] RESUB_COUNT=1: untouched (requeue re-passes its own resolved env)")

            # (c) an explicit value already in cfg is never overwritten
            os.environ.pop("RESUB_COUNT", None)
            cfg3 = {"L": 4, "hx": 0.2, "compute_dtype": "tf32"}
            apply_late_lever_defaults(cfg3, prefix="test")
            assert cfg3["compute_dtype"] == "tf32", cfg3
            assert cfg3.get("inv_impl") == "dense"       # inv_impl was still unset -> filled
            print("[2c] explicit compute_dtype=tf32 wins over the file; unset inv_impl still filled")

            # (d) no file at TC3D_DEFAULTS_FILE -> untouched
            os.environ["TC3D_DEFAULTS_FILE"] = "/dev/null"
            cfg4 = {"L": 4, "hx": 0.2}
            apply_late_lever_defaults(cfg4, prefix="test")
            assert cfg4.get("compute_dtype") is None and cfg4.get("inv_impl") is None
            print("[2d] TC3D_DEFAULTS_FILE=/dev/null: untouched (disabled, same as the wrapper)")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    test_check_resume_config()
    test_late_lever_defaults()
    print("ALL PASSED")
