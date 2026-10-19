"""Copy a directory tree (8.3 names only) into a freshly formatted FAT32
image's root directory. Minimal on purpose: enough to build test fixtures
without mtools."""
import os, struct, sys

img_path, tree = sys.argv[1], sys.argv[2]
f = open(img_path, 'r+b')
bs = f.read(512)
bps, spc = struct.unpack_from('<HB', bs, 11)
rsvd, nfats = struct.unpack_from('<HB', bs, 14)
fatsz, = struct.unpack_from('<I', bs, 36)
root, = struct.unpack_from('<I', bs, 44)
cbytes = bps * spc
fat_off = rsvd * bps
data_off = (rsvd + nfats * fatsz) * bps
nclusters = fatsz * bps // 4
f.seek(fat_off)
fat = bytearray(f.read(fatsz * bps))
next_free = [3]

def alloc_chain(n):
    out = []
    c = next_free[0]
    while len(out) < n:
        if struct.unpack_from('<I', fat, c * 4)[0] & 0x0FFFFFFF == 0:
            out.append(c)
        c += 1
    next_free[0] = c
    for a, b in zip(out, out[1:] + [0x0FFFFFFF]):
        struct.pack_into('<I', fat, a * 4, b)
    return out

def write_cluster(c, data):
    f.seek(data_off + (c - 2) * cbytes)
    f.write(data.ljust(cbytes, b'\0'))

def name83(n):
    base, _, ext = n.upper().partition('.')
    return base.ljust(8)[:8].encode() + ext.ljust(3)[:3].encode()

def entry(name, attr, cluster, size):
    return name83(name) + struct.pack('<BBBHHHHHHHI', attr, 0, 0, 0, 0, 0,
                                      cluster >> 16, 0, 0, cluster & 0xFFFF, size)

def add_dir(path, dir_cluster, parent):
    ents = []
    if parent is not None:
        ents.append(b'.          ' + struct.pack('<BBBHHHHHHHI', 0x10, 0, 0, 0, 0, 0, dir_cluster >> 16, 0, 0, dir_cluster & 0xFFFF, 0))
        ents.append(b'..         ' + struct.pack('<BBBHHHHHHHI', 0x10, 0, 0, 0, 0, 0, parent >> 16, 0, 0, parent & 0xFFFF, 0))
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            c = alloc_chain(1)[0]
            add_dir(p, c, 0 if dir_cluster == root else dir_cluster)
            ents.append(entry(name, 0x10, c, 0))
        else:
            data = open(p, 'rb').read()
            chain = alloc_chain(max(1, -(-len(data) // cbytes)))
            for i, c in enumerate(chain):
                write_cluster(c, data[i * cbytes:(i + 1) * cbytes])
            ents.append(entry(name, 0x20, chain[0], len(data)))
    if dir_cluster == root:
        f.seek(data_off + (root - 2) * cbytes)
        existing = bytearray(f.read(cbytes))   # keep the volume label entry
        off = 0
        while existing[off] not in (0,):
            off += 32
        blob = bytes(existing[:off]) + b''.join(ents)
    else:
        blob = b''.join(ents)
    assert len(blob) <= cbytes
    write_cluster(dir_cluster, blob)

add_dir(tree, root, None)
for i in range(nfats):
    f.seek(fat_off + i * fatsz * bps)
    f.write(fat)
f.close()
