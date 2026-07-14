import pandas as pd, numpy as np

df = pd.read_pickle('/g/data/gv90/xl1657/phd/M3_workspace/output/R3_input_d070_verified.pkl')
wed = df[df['sector']=='Weddell']

eta_s = 1.263
delta = 0.70
rho_w, rho_i, rho_s = 1024.0, 900.0, 330.0

is2_fb = wed['is2_fb_wm'].values
cs2_rfb = wed['cs2_rfb'].values
hs = wed['hs_d070'].values

# Method A: Total freeboard form (Kacimi & Kwok 2020) — CORRECT
h_i_A = (rho_w * is2_fb - (rho_w - rho_s) * hs) / (rho_w - rho_i)

# Method B: Ice freeboard form (Zwally et al. 2008) — CORRECT  
h_fi_B = is2_fb - hs  # ice freeboard = total freeboard minus snow
h_i_B = (rho_w * h_fi_B + rho_s * hs) / (rho_w - rho_i)

# Method C: What Step02 currently does — WRONG coefficient
h_fi_C = cs2_rfb + (1.0 - delta) * hs / eta_s
h_i_C = (rho_w * h_fi_C + (rho_s - rho_w) * hs) / (rho_w - rho_i)

# Method D: Check equivalence of A and B
print("Weddell mean ice thickness:")
print(f"  Method A (total fb, K&K 2020):  {np.mean(h_i_A):.3f} m  ← CORRECT")
print(f"  Method B (ice fb, Zwally 2008): {np.mean(h_i_B):.3f} m  ← CORRECT")
print(f"  Method C (Step02 current):      {np.mean(h_i_C):.3f} m  ← WRONG")
print(f"\n  A-B difference: {np.mean(h_i_A - h_i_B):.6f} m (should be ~0)")
print(f"\n  is2_fb mean:    {np.mean(is2_fb):.4f} m")
print(f"  cs2_rfb mean:   {np.mean(cs2_rfb):.4f} m")
print(f"  delta_f mean:   {np.mean(is2_fb - cs2_rfb):.4f} m")
print(f"  h_s mean:       {np.mean(hs):.4f} m")
print(f"  h_fi mean:      {np.mean(is2_fb - hs):.4f} m")

# Physical plausibility check
print(f"\n  h_i_A valid (>0): {(h_i_A > 0).mean()*100:.1f}%")
print(f"  h_i_A median:     {np.median(h_i_A):.3f} m")
print(f"  h_i_A IQR:        [{np.percentile(h_i_A,25):.3f}, {np.percentile(h_i_A,75):.3f}] m")
