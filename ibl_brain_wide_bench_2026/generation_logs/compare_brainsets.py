# %%
from pathlib import Path

import h5py
import iblutil.io.hashfile
import numpy as np

eid = "0a018f12-ee06-4b11-97aa-bbbff5448e9f"  # eval eids small
# swap in proc_test/ to compare a scratch build instead of the current one
a = {
    "file": next(
        Path("/mnt/s0/BrainSets/proc/all_units/ibl_brain_wide_bench_2026").rglob(f"{eid}*.h5")
    )
}
b = {
    "file": next(
        Path("/mnt/s0/BrainSets/ref/ibl_brain_wide_bench_2026/all_units").rglob(f"{eid}*.h5")
    )
}
print(f"h5diff --delta=1e-8  {a['file']} {b['file']}")

print(iblutil.io.hashfile.md5(a["file"]))
print(iblutil.io.hashfile.md5(b["file"]))

a["fp"] = h5py.File(a["file"], "r")
b["fp"] = h5py.File(b["file"], "r")

print(a["fp"].keys())
print(b["fp"].keys())


def visitor(name, obj):
    if isinstance(obj, h5py.Dataset):
        if obj.dtype != np.dtype("object"):
            lab = "PASS" if np.array_equal(a["fp"][name][:], b["fp"][name][:]) else "CLOSE"
            np.testing.assert_allclose(a["fp"][name][:], b["fp"][name][:], rtol=1e-5)
            print(f"{lab} Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
        else:
            np.testing.assert_array_equal(a["fp"][name][:], b["fp"][name][:])
            print(f"PASS Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")

    elif isinstance(obj, h5py.Group):
        print(f"Group:   {name}")


b["fp"].visititems(visitor)

np.testing.assert_array_equal(a["fp"]["trials/block"][:], b["fp"]["trials/block"][:])


# %%
for attr, _ in a["fp"]["brainset"].attrs.items():
    print(attr, a["fp"]["brainset"].attrs[attr], " ", b["fp"]["brainset"].attrs[attr])


# attribute: <temporaldata_version of </brainset>> and <temporaldata_version of </brainset>>
# 10 differences found
# dataset: </licks/domain/end> and </licks/domain/end>
# 1 differences found
# dataset: </paws/domain/end> and </paws/domain/end>
# 1 differences found
# dataset: </wheel/domain/end> and </wheel/domain/end>
# 1 differences found
# dataset: </whisker/domain/end> and </whisker/domain/end>
