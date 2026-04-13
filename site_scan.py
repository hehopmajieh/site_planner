#!/usr/bin/env python3

import math
import csv
from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import Geod
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DEM_FILE = "output_hh.tif"

CENTER_LAT = 42.1354   # Plovdiv
CENTER_LON = 24.7453

RADIUS_KM = 60.0
GRID_STEP_KM = 2.5

MIN_SITE_ELEV_M = 300.0
ANTENNA_HEIGHT_M = 8.0

# How far to inspect the local horizon from each site
LOOK_DISTANCE_KM = 30.0
PROFILE_STEP_M = 250.0

# Azimuth sector toward West / Northwest Europe
AZIMUTH_START_DEG = 260.0
AZIMUTH_END_DEG = 330.0
AZIMUTH_STEP_DEG = 5.0

geod = Geod(ellps="WGS84")


@dataclass
class SiteResult:
    lat: float
    lon: float
    elev_m: float
    score: float
    worst_horizon_deg: float
    avg_horizon_deg: float
    best_azimuth_deg: float


class DEMSampler:
    def __init__(self, dem_file: str):
        self.ds = rasterio.open(dem_file)
        self.band = self.ds.read(1, masked=True)
        self.bounds = self.ds.bounds

        print("\nDEM info")
        print("--------")
        print("CRS:", self.ds.crs)
        print("Bounds:", self.ds.bounds)
        print("Size:", self.ds.width, "x", self.ds.height)
        print("Resolution:", self.ds.res)
        print("Nodata:", self.ds.nodata)
        print()

    def in_bounds(self, lon: float, lat: float) -> bool:
        b = self.bounds
        return (b.left <= lon <= b.right) and (b.bottom <= lat <= b.top)

    def sample(self, lon: float, lat: float) -> float:
        if not self.in_bounds(lon, lat):
            return np.nan

        try:
            row, col = self.ds.index(lon, lat)
        except Exception:
            return np.nan

        if row < 0 or row >= self.band.shape[0] or col < 0 or col >= self.band.shape[1]:
            return np.nan

        val = self.band[row, col]
        if np.ma.is_masked(val):
            return np.nan

        return float(val)


def destination_point(lat: float, lon: float, azimuth_deg: float, distance_km: float):
    lon2, lat2, _ = geod.fwd(lon, lat, azimuth_deg, distance_km * 1000.0)
    return lat2, lon2


def generate_candidate_grid(center_lat, center_lon, radius_km, step_km):
    pts = []
    n = int((2 * radius_km) / step_km) + 1

    for i in range(n):
        north_km = -radius_km + i * step_km
        for j in range(n):
            east_km = -radius_km + j * step_km

            if math.hypot(north_km, east_km) > radius_km:
                continue

            dist_km = math.hypot(north_km, east_km)
            az = math.degrees(math.atan2(east_km, north_km)) % 360.0
            lat, lon = destination_point(center_lat, center_lon, az, dist_km)
            pts.append((lat, lon))

    return pts


def horizon_along_azimuth(dem, lat, lon, azimuth_deg, look_distance_km, antenna_height_m, step_m):
    site_ground = dem.sample(lon, lat)
    if np.isnan(site_ground):
        return None

    site_abs = site_ground + antenna_height_m

    n = max(2, int((look_distance_km * 1000.0) / step_m) + 1)
    dists = np.linspace(step_m, look_distance_km * 1000.0, n)

    angles = []
    for d in dists:
        lonx, latx, _ = geod.fwd(lon, lat, azimuth_deg, d)
        h = dem.sample(lonx, latx)
        if np.isnan(h):
            return None

        angle_deg = math.degrees(math.atan2(h - site_abs, d))
        angles.append(angle_deg)

    if not angles:
        return None

    angles = np.array(angles, dtype=float)

    return {
        "max_horizon_deg": float(np.max(angles)),
        "p95_horizon_deg": float(np.percentile(angles, 95)),
        "p99_horizon_deg": float(np.percentile(angles, 99)),
    }


def evaluate_site(dem, lat, lon):
    elev = dem.sample(lon, lat)
    if np.isnan(elev) or elev < MIN_SITE_ELEV_M:
        return None

    best_score = -1e9
    best_azimuth = None
    best_worst_h = None
    best_avg_h = None

    azimuths = np.arange(AZIMUTH_START_DEG, AZIMUTH_END_DEG + 0.1, AZIMUTH_STEP_DEG)

    for az in azimuths:
        hm = horizon_along_azimuth(
            dem=dem,
            lat=lat,
            lon=lon,
            azimuth_deg=float(az),
            look_distance_km=LOOK_DISTANCE_KM,
            antenna_height_m=ANTENNA_HEIGHT_M,
            step_m=PROFILE_STEP_M,
        )
        if hm is None:
            continue

        worst_h = hm["p99_horizon_deg"]
        avg_h = hm["p95_horizon_deg"]

        # Lower horizon angle is better
        horizon_score = np.clip((3.0 - worst_h) / 3.0, 0.0, 1.0)
        avg_score = np.clip((2.0 - avg_h) / 2.0, 0.0, 1.0)
        elev_score = np.clip((elev - MIN_SITE_ELEV_M) / 1800.0, 0.0, 1.0)

        score = 0.55 * horizon_score + 0.20 * avg_score + 0.25 * elev_score

        if score > best_score:
            best_score = score
            best_azimuth = float(az)
            best_worst_h = worst_h
            best_avg_h = avg_h

    if best_azimuth is None:
        return None

    return SiteResult(
        lat=lat,
        lon=lon,
        elev_m=float(elev),
        score=float(best_score),
        worst_horizon_deg=float(best_worst_h),
        avg_horizon_deg=float(best_avg_h),
        best_azimuth_deg=float(best_azimuth),
    )


def main():
    dem = DEMSampler(DEM_FILE)
    candidates = generate_candidate_grid(CENTER_LAT, CENTER_LON, RADIUS_KM, GRID_STEP_KM)

    print(f"Generated candidate points: {len(candidates)}")

    valid_dem_points = 0
    high_enough_points = 0
    results = []

    for idx, (lat, lon) in enumerate(candidates, 1):
        elev = dem.sample(lon, lat)

        if not np.isnan(elev):
            valid_dem_points += 1

        if not np.isnan(elev) and elev >= MIN_SITE_ELEV_M:
            high_enough_points += 1

        r = evaluate_site(dem, lat, lon)
        if r is not None:
            results.append(r)

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(candidates)}")

    print()
    print("Debug summary")
    print("-------------")
    print("Valid DEM points:", valid_dem_points)
    print("Points above MIN_SITE_ELEV_M:", high_enough_points)
    print("Final result points:", len(results))
    print()

    results.sort(key=lambda x: x.score, reverse=True)

    print("Top 20 sites:")
    for i, r in enumerate(results[:20], 1):
        print(
            f"{i:2d}. lat={r.lat:.5f}, lon={r.lon:.5f}, "
            f"elev={r.elev_m:.0f} m, score={r.score:.3f}, "
            f"worst_h={r.worst_horizon_deg:.2f}°, avg_h={r.avg_horizon_deg:.2f}°, "
            f"best_az={r.best_azimuth_deg:.0f}°"
        )

    with open("site_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "lat", "lon", "elev_m", "score",
            "worst_horizon_deg", "avg_horizon_deg", "best_azimuth_deg"
        ])
        for r in results:
            w.writerow([
                r.lat, r.lon, r.elev_m, r.score,
                r.worst_horizon_deg, r.avg_horizon_deg, r.best_azimuth_deg
            ])

    plt.figure(figsize=(10, 8))

    if results:
        lons = [r.lon for r in results]
        lats = [r.lat for r in results]
        scores = [r.score for r in results]

        sc = plt.scatter(lons, lats, c=scores, s=22, cmap="viridis")
        plt.colorbar(sc, label="Suitability score")

    plt.scatter([CENTER_LON], [CENTER_LAT], marker="x", s=120, label="Plovdiv")

    b = dem.bounds
    plt.xlim(b.left, b.right)
    plt.ylim(b.bottom, b.top)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Local horizon suitability for West Europe")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("site_map.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
