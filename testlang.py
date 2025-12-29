import h5py, numpy as np

path = "results/SynthText_0001.h5"  # или SynthText_0001.h5 и т.п.

# with h5py.File(path, "r") as f:
#     print("top keys:", list(f.keys()))
#     print("/data count:", len(f["data"]))
#     name = next(iter(f["data"].keys()))
#     d = f["data"][name]
#     print("sample dataset:", name)
#     print("attrs keys:", list(d.attrs.keys()))
#     if "txt" in d.attrs:
#         print("txt sample:", d.attrs["txt"][:5])
#     if "lang" in d.attrs:
#         print("lang sample:", d.attrs["lang"][:5])


with h5py.File(path, "r") as f:
    k = next(iter(f["data"].keys()))
    d = f["data"][k]
    t = d.attrs["txt"]
    print("dtype:", np.array(t).dtype)
    print("repr first 10:", [repr(x) for x in list(t)[:10]])
    print("empty count:", sum((str(x) == "") for x in t), "/", len(t))