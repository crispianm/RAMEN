import json, numpy as np
d = json.load(open(__file__.rsplit('/',1)[0] + '/perbin_raw.json')); B = d['bins']; S = d['sources']
tm = sum(s['mass'] for s in S.values())
def row(vals, tot): return ' | '.join(f'{100*v/tot:.3f}' for v in vals)
print('## Table A: per source, % of ALL pixels of the sampled clips (8-frame clips at 518, 40 clips/source, seed 45)\n')
print('| source | mass share | clips | sky % | valid % | ' + ' | '.join(B) + ' | SiLog-eligible % (all bins) | SiLog-elig >=80 m % |'); print('|' + '---|' * (7 + len(B)))
mix = {'valid': np.zeros(len(B)), 'silog': np.zeros(len(B)), 'sky': 0.0}
for n, s in S.items():
    t = s['total_px']; w = s['mass'] / tm
    v = np.array(s['valid_px']); g = np.array(s['silog_px'])
    print(f"| {n} | {100*w:.1f}% | {s['clips']} | {100*s['sky_px']/t:.2f} | {100*v.sum()/t:.1f} | {row(v, t)} | {100*g.sum()/t:.1f} | {100*g[3:].sum()/t:.3f} |")
    mix['valid'] += w * v / t; mix['silog'] += w * g / t; mix['sky'] += w * s['sky_px'] / t
print('\n## Table B: mass-weighted MIX (expected fraction of all pixels in a sampled batch; weight = mass_s / sum mass)\n')
print('| quantity | ' + ' | '.join(B) + ' | total |'); print('|' + '---|' * (len(B) + 2))
print('| valid GT px (VDA loss mask) % | ' + ' | '.join(f'{100*x:.3f}' for x in mix['valid']) + f' | {100*mix["valid"].sum():.2f} |')
print('| SiLog-eligible px % | ' + ' | '.join(f'{100*x:.3f}' for x in mix['silog']) + f' | {100*mix["silog"].sum():.2f} |')
print('| share of SiLog-eligible px | ' + ' | '.join(f'{100*x/mix["silog"].sum():.2f}' for x in mix['silog']) + ' | 100 |')
print(f'\nsky px (VDA zero-disparity target only) = {100*mix["sky"]:.2f}% of all px')
print('\n## Table C: far (>=80 m) SiLog pixel share of the mix by source\n')
tot = sum(sum(S[n]['silog_px'][3:]) / S[n]['total_px'] * S[n]['mass'] / tm for n in S)
for n, s in S.items():
    f = sum(s['silog_px'][3:]) / s['total_px'] * s['mass'] / tm
    f120 = sum(s['silog_px'][4:6]) / s['total_px'] * s['mass'] / tm
    print(f'- {n}: {100*f:.3f}% of all px = {100*f/tot:.1f}% of far SiLog px;  120-655 m: {100*f120:.3f}%')
print(f'- mix total >=80 m SiLog-eligible: {100*tot:.3f}% of all px = {100*tot/mix["silog"].sum():.2f}% of SiLog px')
