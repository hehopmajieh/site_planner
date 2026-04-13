#!/usr/bin/env python3

"""
Local horizon + Fresnel clearance + diffraction loss scan
for VHF/UHF contest site selection.

This script:
- Loads a local DEM (GeoTIFF, EPSG:4326)
- Generates candidate sites around Plovdiv
- Evaluates local horizon toward a W/NW azimuth sector
- Computes 60% first Fresnel zone clearance
- Computes approximate knife-edge diffraction loss
- Outputs ranked candidate sites
"""

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

LOOK_DISTANCE_KM = 30.0
PROFILE_STEP_M = 250.0

AZIMUTH_START_DEG = 260.0
AZIMUTH_END_DEG = 330.0
AZIMUTH_STEP_DEG = 5.0

FREQ_MHZ = 144.0

EARTH_RADIUS_M = 6371000.0
K_FACTOR = 4.0 / 3.0

geod = Geod(ellps="WGS84")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SiteResult:
    lat: float
    lon: float
    elev_m: float
    score: float
    best_azimuth_deg: float
    worst_horizon_deg: float
    avg_horizon_deg: float
    min_fresnel_clearance_m: float
    avg_fresnel_clearance_m: float
    max_diffraction_loss_db: float
    worst_nu: float


# ============================================================
# DEM HANDLING
# ============================================================

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


# ============================================================
# GEOMETRY / RF UTILS
# ============================================================

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


def fresnel_radius_m(d1_m: float, d2_m: float, freq_mhz: float) -> float:
    freq_hz = freq_mhz * 1e6
    wavelength_m = 3e8 / freq_hz
    return math.sqrt(wavelength_m * d1_m * d2_m / (d1_m + d2_m))


def earth_bulge_m(d1_m: float, d2_m: float, k_factor: float = K_FACTOR) -> float:
    reff = EARTH_RADIUS_M * k_factor
    return (d1_m * d2_m) / (2.0 * reff)


def knife_edge_loss_db(nu: float) -> float:
    """
    ITU-style single knife-edge approximation.
    """
    if nu <= -0.78:
        return 0.0
    return 6.9 + 20.0 * math.log10(
        math.sqrt((nu - 0.1) ** 2 + 1.0) + nu - 0.1
    )


# ============================================================
# PROFILE ANALYSIS
# ============================================================

def sample_along_azimuth(dem, lat, lon, azimuth_deg, look_distance_km, step_m):
    total_m = look_distance_km * 1000.0
    n = max(2, int(total_m / step_m) + 1)
    dists = np.linspace(0.0, total_m, n)

    elev = []
    valid = True

    for d in dists:
        lonx, latx, _ = geod.fwd(lon, lat, azimuth_deg, d)
        h = dem.sample(lonx, latx)
        elev.append(h)
        if np.isnan(h):
            valid = False

    return dists, np.array(elev, dtype=float), valid


def evaluate_azimuth(dem, lat, lon, azimuth_deg):
    dists, terr, valid = sample_along_azimuth(
        dem=dem,
        lat=lat,
        lon=lon,
        azimuth_deg=azimuth_deg,
        look_distance_km=LOOK_DISTANCE_KM,
        step_m=PROFILE_STEP_M,
    )

    if not valid or len(dists) < 3:
        return None

    site_ground = terr[0]
    if np.isnan(site_ground):
        return None

    tx_abs = site_ground + ANTENNA_HEIGHT_M
    rx_abs = terr[-1] + ANTENNA_HEIGHT_M

    total_m = dists[-1]
    los = tx_abs + (rx_abs - tx_abs) * (dists / total_m)

    horizon_angles = []
    fresnel_clearances = []
    nus = []
    losses = []

    for i in range(1, len(dists) - 1):
        d = dists[i]

        if d < 500.0:
            continue

        d1 = d
        d2 = total_m - d

        bulge = earth_bulge_m(d1, d2, K_FACTOR)
        effective_terrain = terr[i] + bulge

        # Horizon angle
        angle_deg = math.degrees(math.atan2(effective_terrain - tx_abs, d1))
        horizon_angles.append(angle_deg)

        # Fresnel
        fr = fresnel_radius_m(d1, d2, FREQ_MHZ)
        clearance = los[i] - effective_terrain - 0.6 * fr
        fresnel_clearances.append(clearance)

        # Knife-edge diffraction parameter nu
        # h is positive if obstacle is above LOS
        h = effective_terrain - los[i]
        nu = math.sqrt(2.0) * h / fr
        nus.append(nu)

        loss_db = knife_edge_loss_db(nu)
        losses.append(loss_db)

    if not horizon_angles or not fresnel_clearances or not losses:
        return None

    horizon_angles = np.array(horizon_angles, dtype=float)
    fresnel_clearances = np.array(fresnel_clearances, dtype=float)
    nus = np.array(nus, dtype=float)
    losses = np.array(losses, dtype=float)

    return {
        "worst_horizon_deg": float(np.percentile(horizon_angles, 99)),
        "avg_horizon_deg": float(np.percentile(horizon_angles, 95)),
        "min_fresnel_clearance_m": float(np.min(fresnel_clearances)),
        "avg_fresnel_clearance_m": float(np.mean(fresnel_clearances)),
        "max_diffraction_loss_db": float(np.max(losses)),
        "worst_nu": float(np.max(nus)),
    }


# ============================================================
# SITE EVALUATION
# ============================================================

def evaluate_site(dem, lat, lon):
    elev = dem.sample(lon, lat)
    if np.isnan(elev) or elev < MIN_SITE_ELEV_M:
        return None

    azimuths = np.arange(AZIMUTH_START_DEG, AZIMUTH_END_DEG + 0.1, AZIMUTH_STEP_DEG)

    best_score = -1e9
    best_result = None

    for az in azimuths:
        res = evaluate_azimuth(dem, lat, lon, float(az))
        if res is None:
            continue

        worst_h = res["worst_horizon_deg"]
        avg_h = res["avg_horizon_deg"]
        min_fc = res["min_fresnel_clearance_m"]
        avg_fc = res["avg_fresnel_clearance_m"]
        max_loss = res["max_diffraction_loss_db"]
        worst_nu = res["worst_nu"]

        horizon_score = np.clip((3.0 - worst_h) / 3.0, 0.0, 1.0)
        avg_h_score = np.clip((2.0 - avg_h) / 2.0, 0.0, 1.0)

        min_fc_score = np.clip((min_fc + 100.0) / 200.0, 0.0, 1.0)
        avg_fc_score = np.clip((avg_fc + 100.0) / 200.0, 0.0, 1.0)

        # Lower diffraction loss is better
        diff_score = np.clip((20.0 - max_loss) / 20.0, 0.0, 1.0)

        elev_score = np.clip((elev - MIN_SITE_ELEV_M) / 1800.0, 0.0, 1.0)

        score = (
            0.22 * horizon_score +
            0.10 * avg_h_score +
            0.23 * min_fc_score +
            0.10 * avg_fc_score +
            0.25 * diff_score +
            0.10 * elev_score
        )

        if score > best_score:
            best_score = score
            best_result = SiteResult(
                lat=lat,
                lon=lon,
                elev_m=float(elev),
                score=float(score),
                best_azimuth_deg=float(az),
                worst_horizon_deg=float(worst_h),
                avg_horizon_deg=float(avg_h),
                min_fresnel_clearance_m=float(min_fc),
                avg_fresnel_clearance_m=float(avg_fc),
                max_diffraction_loss_db=float(max_loss),
                worst_nu=float(worst_nu),
            )

    return best_result


# ============================================================
# MAIN
# ============================================================

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
            f"worst_h={r.worst_horizon_deg:.2f}°, "
            f"min_fc={r.min_fresnel_clearance_m:.1f} m, "
            f"diff={r.max_diffraction_loss_db:.1f} dB, "
            f"nu={r.worst_nu:.2f}, "
            f"best_az={r.best_azimuth_deg:.0f}°"
        )

    with open("site_candidates_fresnel_diffraction.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "lat", "lon", "elev_m", "score",
            "best_azimuth_deg",
            "worst_horizon_deg", "avg_horizon_deg",
            "min_fresnel_clearance_m", "avg_fresnel_clearance_m",
            "max_diffraction_loss_db", "worst_nu"
        ])
        for r in results:
            w.writerow([
                r.lat, r.lon, r.elev_m, r.score,
                r.best_azimuth_deg,
                r.worst_horizon_deg, r.avg_horizon_deg,
                r.min_fresnel_clearance_m, r.avg_fresnel_clearance_m,
                r.max_diffraction_loss_db, r.worst_nu
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
    plt.title(f"Local horizon + Fresnel + diffraction ({FREQ_MHZ:.0f} MHz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("site_map_fresnel_diffraction.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
