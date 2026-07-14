import pandas as pd
import numpy as np

df = pd.read_pickle('/g/data/gv90/xl1657/phd/M3_workspace/output/R3_input_d070_verified.pkl')

for sector in ['Weddell', 'Ross']:
    mask = df['sector'] == sector
    hs = df.loc[mask, 'hs_d070']
    valid = hs > 0
    
    print(f"\n{sector} (n_all={mask.sum()}, n_valid={valid.sum()}, neg={100*(~valid).mean():.1f}%):")
    print(f"  ALL matchups:   mean={hs.mean():.3f}  median={hs.median():.3f}")
    print(f"  VALID only:     mean={hs[valid].mean():.3f}  median={hs[valid].median():.3f}")
    print(f"  M2 manuscript:  mean=??? median=???  (should match VALID only)")

print("\nExpected M2 values (δ=0.70, valid only):")
print("  Weddell: mean 0.255, median 0.188")
print("  Ross:    mean 0.217, median 0.156")
