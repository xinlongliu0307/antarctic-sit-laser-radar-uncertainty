import pandas as pd, numpy as np

df = pd.read_pickle('/g/data/gv90/xl1657/phd/M3_workspace/output/R3_input_d070_verified.pkl')
wed = df[df['sector']=='Weddell']

eta_s = 1.263
delta = 0.70

# Method 1: IS2 freeboard path (no explicit delta — penetration already in h_s)
h_fi_is2 = wed['is2_fb_wm'] - wed['hs_d070'] / eta_s
h_i_is2 = (1024 * h_fi_is2 + (330 - 1024) * wed['hs_d070']) / (1024 - 900)

# Method 2: CS2 radar freeboard path with delta correction (uses h_s at delta=1.0)
hs_d10 = wed['snow_thickness']  # CSV column, at delta=1.0
h_fi_cs2 = wed['cs2_rfb'] + (1 - delta) * hs_d10 / eta_s
h_i_cs2 = (1024 * h_fi_cs2 + (330 - 1024) * wed['hs_d070']) / (1024 - 900)

# Method 3: WRONG — what Step02 currently does (double-correction)
h_fi_wrong = wed['cs2_rfb'] + (1 - delta) * wed['hs_d070'] / eta_s
h_i_wrong = (1024 * h_fi_wrong + (330 - 1024) * wed['hs_d070']) / (1024 - 900)

print("Weddell mean ice thickness:")
print(f"  Method 1 (IS2 path, correct):    {h_i_is2.mean():.3f} m")
print(f"  Method 2 (CS2 + delta=1 hs):     {h_i_cs2.mean():.3f} m")  
print(f"  Method 3 (CS2 + delta=0.7 hs):   {h_i_wrong.mean():.3f} m  ← Step02 current")
print(f"\n  h_fi_is2 mean:   {h_fi_is2.mean():.4f} m")
print(f"  h_fi_cs2 mean:   {h_fi_cs2.mean():.4f} m")
print(f"  h_fi_wrong mean: {h_fi_wrong.mean():.4f} m")
