import math
import csv
from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import Geod
import matplotlib.pyplot as plt


DEM_FILE = "output_hh.tif"

CENTER_LAT = 42.1354   # Plovdiv
CENTER_LON = 24.7453
RADIUS_KM = 100.0

GRID_STEP_KM = 2.5
PROFILE_STEP_M = 250.0
MIN_SITE_ELEV_M = 700.0
ANTENNA_HEIGHT_M = 8.0

# Западна Европа – representative направления
TARGETS = [
    ("Zagreb",   45.8150, 15.9819),
    ("Ljubljana",46.0569, 14.5058),
    ("Vienna",   48.2082, 16.3738),
    ("Munich",   48.1351, 11.5820),
    ("Milan",    45.4642,  9.1900),
]

geod = Geod(ellps="WGS84")


@dataclass
class SiteResult:
    lat: float
    lon: float
    elev_m: float
    score: float
    max_horizon_deg: float
    avg_horizon_deg: float
    best_target: str


class DEMSampler:
    def __init__(self, dem_file: str):
        self.ds = rasterio.open(dem_file)
        self.band = self.ds.read(1, masked=True)

    def sample(self, lon: float, lat: float) -> float:
        row, col = self.ds.index(lon, lat)
        if row < 0 or row >= self.band.shape[0] or col < 0 or col >= self.band.shape[1]:
            return np.nan
        val = self.band[row, col]
        if np.ma.is_masked(val):
            return np.nan
        return float(val)


def destination_point(lat: float, lon: float, az_deg: float, dist_km: float):
    lon2, lat2, _ = geod.fwd(lon, lat, az_deg, dist_km * 1000.0)
    return lat2, lon2


def distance_azimuth(lat1: float, lon1: float, lat2: float, lon2: float):
    az12, az21, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m, az12


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


def terrain_profile(dem, lat1, lon1, lat2, lon2, step_m=250.0):
    dist_m, az = distance_azimuth(lat1, lon1, lat2, lon2)
    n = max(2, int(dist_m / step_m) + 1)
    dists = np.linspace(0, dist_m, n)

    terr = []
    for d in dists:
        lonx, latx, _ = geod.fwd(lon1, lat1, az, d)
        terr.append(dem.sample(lonx, latx))

    return dists, np.array(terr)


def horizon_metrics(dem, tx_lat, tx_lon, rx_lat, rx_lon, antenna_height_m, profile_step_m):
    dists, terr = terrain_profile(dem, tx_lat, tx_lon, rx_lat, rx_lon, profile_step_m)
    if np.any(np.isnan(terr)):
        return None

    tx_abs = terr[0] + antenna_height_m
    angles = []

    # гледаме всички междинни точки
    for i in range(1, len(dists) - 1):
        dh = terr[i] - tx_abs
        angle_deg = math.degrees(math.atan2(dh, dists[i]))
        angles.append(angle_deg)

    if not angles:
        return None

    return {
        "max_horizon_deg": float(np.max(angles)),
        "avg_horizon_deg": float(np.mean(np.sort(angles)[-10:] if len(angles) >= 10 else angles))
    }


def evaluate_site(dem, lat, lon):
    elev = dem.sample(lon, lat)
    if np.isnan(elev) or elev < MIN_SITE_ELEV_M:
        return None

    best_score = -1e9
    best_target = None
    best_max_h = None
    best_avg_h = None

    for name, tlat, tlon in TARGETS:
        hm = horizon_metrics(
            dem,
            lat, lon,
            tlat, tlon,
            ANTENNA_HEIGHT_M,
            PROFILE_STEP_M
        )
        if hm is None:
            continue

        max_h = hm["max_horizon_deg"]
        avg_h = hm["avg_horizon_deg"]

        # по-нисък хоризонт = по-добре
        horizon_score = np.clip((2.0 - max_h) / 2.0, 0, 1)
        avg_score = np.clip((1.5 - avg_h) / 1.5, 0, 1)
        elev_score = np.clip((elev - MIN_SITE_ELEV_M) / 1500.0, 0, 1)

        score = 0.55 * horizon_score + 0.25 * avg_score + 0.20 * elev_score

        if score > best_score:
            best_score = score
            best_target = name
            best_max_h = max_h
            best_avg_h = avg_h

    if best_target is None:
        return None

    return SiteResult(
        lat=lat,
        lon=lon,
        elev_m=elev,
        score=best_score,
        max_horizon_deg=best_max_h,
        avg_horizon_deg=best_avg_h,
        best_target=best_target
    )


def main():
    dem = DEMSampler(DEM_FILE)
    candidates = generate_candidate_grid(CENTER_LAT, CENTER_LON, RADIUS_KM, GRID_STEP_KM)

    print(f"Candidates: {len(candidates)}")

    results = []
    for idx, (lat, lon) in enumerate(candidates, 1):
        r = evaluate_site(dem, lat, lon)
        if r:
            results.append(r)

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(candidates)}")

    results.sort(key=lambda x: x.score, reverse=True)

    print("\nTop 20 sites:")
    for i, r in enumerate(results[:20], 1):
        print(
            f"{i:2d}. lat={r.lat:.5f}, lon={r.lon:.5f}, "
            f"elev={r.elev_m:.0f} m, score={r.score:.3f}, "
            f"max_h={r.max_horizon_deg:.2f}°, avg_h={r.avg_horizon_deg:.2f}°, "
            f"best={r.best_target}"
        )

    with open("site_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "elev_m", "score", "max_horizon_deg", "avg_horizon_deg", "best_target"])
        for r in results:
            w.writerow([r.lat, r.lon, r.elev_m, r.score, r.max_horizon_deg, r.avg_horizon_deg, r.best_target])

    lons = [r.lon for r in results]
    lats = [r.lat for r in results]
    scores = [r.score for r in results]

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(lons, lats, c=scores, s=18, cmap="viridis")
    plt.colorbar(sc, label="Suitability score")
    plt.scatter([CENTER_LON], [CENTER_LAT], marker="x", s=100, label="Plovdiv")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Candidate sites around Plovdiv for West Europe")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("site_suitability_map.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
