from common import *
SEL = {'vkitti': [46, 13, 15, 36, 25, 9, 37, 57, 17, 63], 'tartanair': [1013, 9548, 9551, 1028, 5305, 9246, 9528, 2028, 976, 8779]}
for kind, idx in SEL.items():
    ds = make_source(kind)
    items = [ds[i] for i in idx]
    it = items[0]
    print(kind, {k: (tuple(v.shape), v.dtype) if torch.is_tensor(v) else v for k, v in it.items()})
    if 'degraded_frames' in it:
        print(' degraded-vs-clean max abs diff', float((it['degraded_frames'] - it['frames']).abs().max()))
    torch.save(dict(idx=idx, items=items), SCR / f'batches_{kind}.pt')
