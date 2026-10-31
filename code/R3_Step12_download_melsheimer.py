#!/usr/bin/env python3
"""
Download University of Bremen ASCAT+AMSR2 Antarctic sea-ice type products
(Melsheimer et al., 2023) to the M3 workspace on NCI Gadi.

Two products are fetched:
  raw   : ECICE-IcetypesUncorrected-YYYYMMDD.nc  (~19 MB/day; OW, YI, FYI, MYI,
          TOTAL_ICE in %, confidences, LAT/LON)
  final : MultiYearIce-Antarctic-YYYYMMDD.nc      (~0.2 MB/day; drift-corrected
          MYI, CR-FLAG, EXMYI; no LAT/LON -> take from any raw file)

Both are on the 664x632 NSIDC 12.5 km south polar stereographic grid, EPSG:3412.

Layout created under --root (default: M3_workspace):
  data/melsheimer/raw/YYYY/ECICE-IcetypesUncorrected-YYYYMMDD.nc
  data/melsheimer/final/YYYY/MultiYearIce-Antarctic-YYYYMMDD.nc
  output/melsheimer_manifest.csv      one row per file: product, date, path, bytes, status
  logs/download_melsheimer_<stamp>.log

Standard library only (urllib), so it runs with any python3 module on Gadi.
Re-runnable: files already present with the correct size are skipped.

Usage on Gadi (login node or copyq job):
  python3 download_melsheimer_icetype.py --years 2022 2023 2024 2025 --months 4 10
  python3 download_melsheimer_icetype.py --products final --years 2022 2023 2024 2025 --months 1 12
  python3 download_melsheimer_icetype.py --verify          # open every file with netCDF4
  python3 download_melsheimer_icetype.py --dry-run         # list what would be fetched
"""
import argparse
import csv
import datetime as dt
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R3_config_v4 import M3_BASE, MELSHEIMER_YEARS
DEFAULT_ROOT = M3_BASE

BASE = "https://data.seaice.uni-bremen.de/MultiYearIce/ascat-amsr2"
PRODUCTS = {
    "raw":   {"url": BASE + "/raw/Antarctic/netcdf/{year}/",
              "pattern": r"ECICE-IcetypesUncorrected-(\d{8})\.nc"},
    "final": {"url": BASE + "/final/Antarctic/netcdf/{year}/",
              "pattern": r"MultiYearIce-Antarctic-(\d{8})\.nc"},
}
DEFAULT_ROOT = "/g/data/gv90/xl1657/phd/M3_workspace"
USER_AGENT = "M3-icetype-download/1.0 (xinlong.liu@utas.edu.au; UTAS/IMAS research use)"
EXPECTED_SHAPE = (664, 632)


# --------------------------------------------------------------------------- #
def opener():
    op = urllib.request.build_opener()
    op.addheaders = [("User-Agent", USER_AGENT)]
    return op


def fetch(url, retries=4, timeout=120):
    """GET with retries; returns bytes. Raises on final failure."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            with opener().open(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last = e
            logging.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            time.sleep(5 * attempt)
    raise RuntimeError(f"giving up on {url}: {last}")


def remote_size(url, timeout=60):
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def list_index(product, year):
    """Parse the Apache index page -> {date: filename}. Handles missing days."""
    url = PRODUCTS[product]["url"].format(year=year)
    html = fetch(url).decode("utf-8", "replace")
    pat = re.compile(PRODUCTS[product]["pattern"])
    out = {}
    for m in pat.finditer(html):
        d = dt.date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
        out[d] = m.group(0)
    logging.info("%s %d: %d files listed on server", product, year, len(out))
    return url, out


def download_one(url, dest, pause):
    """Stream to dest.part then rename; skip if dest exists with matching size."""
    size = remote_size(url)
    if os.path.exists(dest) and size is not None and os.path.getsize(dest) == size:
        return "skipped", size
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, 5):
        try:
            with opener().open(url, timeout=300) as r, open(part, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
            got = os.path.getsize(part)
            if size is not None and got != size:
                raise IOError(f"size mismatch {got} != {size}")
            os.replace(part, dest)
            time.sleep(pause)
            return "downloaded", got
        except Exception as e:
            logging.warning("attempt %d failed for %s: %s", attempt, url, e)
            time.sleep(10 * attempt)
    if os.path.exists(part):
        os.remove(part)
    return "failed", 0


# --------------------------------------------------------------------------- #
def run_download(args):
    rows = []

    def in_window(d):
        return (args.start <= d <= args.end
                and args.months[0] <= d.month <= args.months[1])

    for product in args.products:
        for year in args.years:
            base_url, listed = list_index(product, year)
            wanted = sorted(d for d in listed if in_window(d))
            missing_days = [d for d in daterange(year, args.months)
                            if in_window(d) and d not in listed]
            if missing_days:
                logging.info("%s %d: %d days absent on server in window (first: %s)",
                             product, year, len(missing_days), missing_days[0])
            logging.info("%s %d: %d files to fetch", product, year, len(wanted))
            for d in wanted:
                fname = listed[d]
                dest = os.path.join(args.root, "data", "melsheimer", product, str(year), fname)
                url = base_url + fname
                if args.dry_run:
                    status, nbytes = "dry-run", remote_size(url) or 0
                else:
                    status, nbytes = download_one(url, dest, args.pause)
                logging.info("%-10s %s %s %10d B", status, product, d, nbytes)
                rows.append(dict(product=product, date=d.isoformat(), file=dest,
                                 bytes=nbytes, status=status))

    write_manifest(args, rows)
    n_fail = sum(r["status"] == "failed" for r in rows)
    logging.info("done: %d files, %d failed", len(rows), n_fail)
    return 1 if n_fail else 0


def daterange(year, months):
    d = dt.date(year, months[0], 1)
    end = dt.date(year + (months[1] == 12), (months[1] % 12) + 1, 1)
    while d < end:
        yield d
        d += dt.timedelta(days=1)


def write_manifest(args, rows):
    os.makedirs(os.path.join(args.root, "output"), exist_ok=True)
    path = os.path.join(args.root, "output", "melsheimer_manifest.csv")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                existing[(r["product"], r["date"])] = r
    for r in rows:
        existing[(r["product"], r["date"])] = r
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["product", "date", "file", "bytes", "status"])
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])
    logging.info("manifest: %s (%d rows)", path, len(existing))


# --------------------------------------------------------------------------- #
def run_verify(args):
    """Open every downloaded file; check grid shape and key variables."""
    try:
        import netCDF4
    except ImportError:
        logging.error("netCDF4 not importable; on Gadi: module load python3/3.10.4 netcdf")
        return 2
    need = {"raw": ["LAT", "LON", "OW", "YI", "FYI", "MYI", "TOTAL_ICE"],
            "final": ["MYI", "CR-FLAG", "EXMYI"]}
    bad = 0
    total = 0
    for product in args.products:
        for year in args.years:
            d = os.path.join(args.root, "data", "melsheimer", product, str(year))
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".nc"):
                    continue
                total += 1
                p = os.path.join(d, fn)
                try:
                    with netCDF4.Dataset(p) as ds:
                        miss = [v for v in need[product] if v not in ds.variables]
                        shp = ds.variables[need[product][-1]].shape
                        if miss or tuple(shp) != EXPECTED_SHAPE:
                            raise ValueError(f"missing={miss} shape={shp}")
                except Exception as e:
                    bad += 1
                    logging.error("BAD %s: %s", p, e)
    logging.info("verify: %d files checked, %d bad", total, bad)
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--products", nargs="+", default=["final", "raw"], choices=list(PRODUCTS))
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--months", nargs=2, type=int, default=[5, 10], metavar=("FIRST", "LAST"),
                    help="inclusive month range, default 5 10 (austral winter matchups)")
    ap.add_argument("--start", default="2022-08-01", help="first date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--end",   default="2025-09-30", help="last date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between files (be polite)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only verify existing files")
    args = ap.parse_args()
    args.start = dt.date.fromisoformat(args.start)
    args.end = dt.date.fromisoformat(args.end)
    args.years = [y for y in args.years if args.start.year <= y <= args.end.year]

    os.makedirs(os.path.join(args.root, "logs"), exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(args.root, "logs", f"download_melsheimer_{stamp}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)])
    logging.info("args: %s", vars(args))
    return run_verify(args) if args.verify else run_download(args)


if __name__ == "__main__":
    sys.exit(main())
