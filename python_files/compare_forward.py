import numpy as np
y_ref = np.fromfile('test_forward_ref.bin', dtype=np.int8).reshape(320, 544)
y_c   = np.fromfile('test_forward_c.bin',   dtype=np.int8).reshape(320, 544)
d = y_ref.astype(np.int32) - y_c.astype(np.int32)
ad = np.abs(d)
print(f"max |diff| : {ad.max()}")
print(f"exatos    : {(d==0).sum()}/{d.size} = {100.0*(d==0).sum()/d.size:.2f}%")
print(f"|diff| ≤ 1: {(ad<=1).sum()}/{d.size} = {100.0*(ad<=1).sum()/d.size:.2f}%")
