#!/usr/bin/env python3

"""
Local horizon + Fresnel clearance + diffraction loss + azimuth coverage scan.

This version:
- reads configuration from JSON
- supports frequency override from CLI
- evaluates all azimuths in the configured sector
- computes:
    * best azimuth
    * number of good azimuths
    * total usable azimuth width
    * best contiguous usable width
- ranks sites using both RF quality and sector usability
"""

import math
import csv
import json
import argparse
from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import Geod
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan candidate sites using local horizon, Fresnel clearance, diffraction and azimuth coverage."
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=None,
        help="Override frequency from config.json"
    )
    return parser.parse_args()


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
    good_azimuth_count: int
    usable_azimuth_width_deg: float
    best_contiguous_width_deg: float


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


class RFScanner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.geod = Geod(ellps="WGS84")

        self.dem_file = cfg["dem_file"]

        self.center_lat = cfg["center"]["lat"]
        self.center_lon = cfg["center"]["lon"]
        self.center_name = cfg["center"].get("name", "Center")

        self.radius_km = cfg["scan"]["radius_km"]
        self.grid_step_km = cfg["scan"]["grid_step_km"]
        self.min_site_elev_m = cfg["scan"]["min_site_elev_m"]
        self.antenna_height_m = cfg["scan"]["antenna_height_m"]

        self.look_distance_km = cfg["sector"]["look_distance_km"]
        self.profile_step_m = cfg["sector"]["profile_step_m"]
        self.azimuth_start_deg = cfg["sector"]["azimuth_start_deg"]
        self.azimuth_end_deg = cfg["sector"]["azimuth_end_deg"]
        self.azimuth_step_deg = cfg["sector"]["azimuth_step_deg"]

        self.freq_mhz = cfg["radio"]["freq_mhz"]
        self.k_factor = cfg["radio"]["k_factor"]

        self.earth_radius_m = cfg["earth"]["radius_m"]

        self.good_min_fc = cfg["coverage"]["good_min_fresnel_clearance_m"]
        self.good_max_diff_db = cfg["coverage"]["good_max_diffraction_loss_db"]

        self.csv_file = cfg["output"]["csv_file"]
        self.plot_file = cfg["output"]["plot_file"]
        self.plot_title = cfg["output"]["plot_title"]

        self.dem = DEMSampler(self.dem_file)

    def destination_point(self, lat: float, lon: float, azimuth_deg: float, distance_km: float):
        lon2, lat2, _ = self.geod.fwd(lon, lat, azimuth_deg, distance_km * 1000.0)
        return lat2, lon2

    def generate_candidate_grid(self):
        pts = []
        n = int((2 * self.radius_km) / self.grid_step_km) + 1

        for i in range(n):
            north_km = -self.radius_km + i * self.grid_step_km
            for j in range(n):
                east_km = -self.radius_km + j * self.grid_step_km

                if math.hypot(north_km, east_km) > self.radius_km:
                    continue

                dist_km = math.hypot(north_km, east_km)
                az = math.degrees(math.atan2(east_km, north_km)) % 360.0
                lat, lon = self.destination_point(self.center_lat, self.center_lon, az, dist_km)
                pts.append((lat, lon))

        return pts

    def fresnel_radius_m(self, d1_m: float, d2_m: float) -> float:
        freq_hz = self.freq_mhz * 1e6
        wavelength_m = 3e8 / freq_hz
        return math.sqrt(wavelength_m * d1_m * d2_m / (d1_m + d2_m))

    def earth_bulge_m(self, d1_m: float, d2_m: float) -> float:
        reff = self.earth_radius_m * self.k_factor
        return (d1_m * d2_m) / (2.0 * reff)

    @staticmethod
    def knife_edge_loss_db(nu: float) -> float:
        if nu <= -0.78:
            return 0.0
        return 6.9 + 20.0 * math.log10(
            math.sqrt((nu - 0.1) ** 2 + 1.0) + nu - 0.1
        )

    def sample_along_azimuth(self, lat: float, lon: float, azimuth_deg: float):
        total_m = self.look_distance_km * 1000.0
        n = max(2, int(total_m / self.profile_step_m) + 1)
        dists = np.linspace(0.0, total_m, n)

        elev = []
        valid = True

        for d in dists:
            lonx, latx, _ = self.geod.fwd(lon, lat, azimuth_deg, d)
            h = self.dem.sample(lonx, latx)
            elev.append(h)
            if np.isnan(h):
                valid = False

        return dists, np.array(elev, dtype=float), valid

    def evaluate_azimuth(self, lat: float, lon: float, azimuth_deg: float):
        dists, terr, valid = self.sample_along_azimuth(lat, lon, azimuth_deg)

        if not valid or len(dists) < 3:
            return None

        site_ground = terr[0]
        if np.isnan(site_ground):
            return None

        tx_abs = site_ground + self.antenna_height_m
        rx_abs = terr[-1] + self.antenna_height_m

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

            bulge = self.earth_bulge_m(d1, d2)
            effective_terrain = terr[i] + bulge

            angle_deg = math.degrees(math.atan2(effective_terrain - tx_abs, d1))
            horizon_angles.append(angle_deg)

            fr = self.fresnel_radius_m(d1, d2)
            clearance = los[i] - effective_terrain - 0.6 * fr
            fresnel_clearances.append(clearance)

            h = effective_terrain - los[i]
            nu = math.sqrt(2.0) * h / fr
            nus.append(nu)

            loss_db = self.knife_edge_loss_db(nu)
            losses.append(loss_db)

        if not horizon_angles or not fresnel_clearances or not losses:
            return None

        horizon_angles = np.array(horizon_angles, dtype=float)
        fresnel_clearances = np.array(fresnel_clearances, dtype=float)
        nus = np.array(nus, dtype=float)
        losses = np.array(losses, dtype=float)

        min_fc = float(np.min(fresnel_clearances))
        max_loss = float(np.max(losses))

        is_good = (min_fc > self.good_min_fc) and (max_loss <= self.good_max_diff_db)

        return {
            "worst_horizon_deg": float(np.percentile(horizon_angles, 99)),
            "avg_horizon_deg": float(np.percentile(horizon_angles, 95)),
            "min_fresnel_clearance_m": min_fc,
            "avg_fresnel_clearance_m": float(np.mean(fresnel_clearances)),
            "max_diffraction_loss_db": max_loss,
            "worst_nu": float(np.max(nus)),
            "is_good": is_good
        }

    def contiguous_width_deg(self, good_flags):
        best_run = 0
        current_run = 0

        for flag in good_flags:
            if flag:
                current_run += 1
                best_run = max(best_run, current_run)
            else:
                current_run = 0

        return best_run * self.azimuth_step_deg

    def evaluate_site(self, lat: float, lon: float):
        elev = self.dem.sample(lon, lat)
        if np.isnan(elev) or elev < self.min_site_elev_m:
            return None

        azimuths = np.arange(
            self.azimuth_start_deg,
            self.azimuth_end_deg + 0.1,
            self.azimuth_step_deg
        )

        az_results = []
        for az in azimuths:
            res = self.evaluate_azimuth(lat, lon, float(az))
            if res is None:
                continue
            az_results.append((float(az), res))

        if not az_results:
            return None

        good_flags = [res["is_good"] for _, res in az_results]
        good_count = sum(good_flags)
        usable_width_deg = good_count * self.azimuth_step_deg
        best_contig_deg = self.contiguous_width_deg(good_flags)

        best_score = -1e9
        best_result = None

        for az, res in az_results:
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
            diff_score = np.clip((20.0 - max_loss) / 20.0, 0.0, 1.0)
            elev_score = np.clip((elev - self.min_site_elev_m) / 1800.0, 0.0, 1.0)

            total_sector_width = (self.azimuth_end_deg - self.azimuth_start_deg)
            usable_width_score = np.clip(usable_width_deg / total_sector_width, 0.0, 1.0)
            contig_width_score = np.clip(best_contig_deg / total_sector_width, 0.0, 1.0)

            score = (
                0.17 * horizon_score +
                0.08 * avg_h_score +
                0.18 * min_fc_score +
                0.08 * avg_fc_score +
                0.18 * diff_score +
                0.08 * elev_score +
                0.11 * usable_width_score +
                0.12 * contig_width_score
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
                    good_azimuth_count=int(good_count),
                    usable_azimuth_width_deg=float(usable_width_deg),
                    best_contiguous_width_deg=float(best_contig_deg),
                )

        return best_result

    def run(self):
        candidates = self.generate_candidate_grid()
        print(f"Generated candidate points: {len(candidates)}")
        print(f"Frequency: {self.freq_mhz:.3f} MHz")

        valid_dem_points = 0
        high_enough_points = 0
        results = []

        for idx, (lat, lon) in enumerate(candidates, 1):
            elev = self.dem.sample(lon, lat)

            if not np.isnan(elev):
                valid_dem_points += 1
            if not np.isnan(elev) and elev >= self.min_site_elev_m:
                high_enough_points += 1

            r = self.evaluate_site(lat, lon)
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
                f"best_az={r.best_azimuth_deg:.0f}°, "
                f"min_fc={r.min_fresnel_clearance_m:.1f} m, "
                f"diff={r.max_diffraction_loss_db:.1f} dB, "
                f"usable={r.usable_azimuth_width_deg:.0f}°, "
                f"contig={r.best_contiguous_width_deg:.0f}°"
            )

        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "lat", "lon", "elev_m", "score",
                "best_azimuth_deg",
                "worst_horizon_deg", "avg_horizon_deg",
                "min_fresnel_clearance_m", "avg_fresnel_clearance_m",
                "max_diffraction_loss_db", "worst_nu",
                "good_azimuth_count",
                "usable_azimuth_width_deg",
                "best_contiguous_width_deg"
            ])
            for r in results:
                w.writerow([
                    r.lat, r.lon, r.elev_m, r.score,
                    r.best_azimuth_deg,
                    r.worst_horizon_deg, r.avg_horizon_deg,
                    r.min_fresnel_clearance_m, r.avg_fresnel_clearance_m,
                    r.max_diffraction_loss_db, r.worst_nu,
                    r.good_azimuth_count,
                    r.usable_azimuth_width_deg,
                    r.best_contiguous_width_deg
                ])

        plt.figure(figsize=(10, 8))

        if results:
            lons = [r.lon for r in results]
            lats = [r.lat for r in results]
            scores = [r.score for r in results]

            sc = plt.scatter(lons, lats, c=scores, s=22, cmap="viridis")
            plt.colorbar(sc, label="Suitability score")

        plt.scatter([self.center_lon], [self.center_lat], marker="x", s=120, label=self.center_name)

        b = self.dem.bounds
        plt.xlim(b.left, b.right)
        plt.ylim(b.bottom, b.top)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{self.plot_title} ({self.freq_mhz:.0f} MHz)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=180)
        plt.show()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.freq_mhz is not None:
        cfg["radio"]["freq_mhz"] = float(args.freq_mhz)

        base_csv = cfg["output"]["csv_file"]
        base_plot = cfg["output"]["plot_file"]

        if base_csv.endswith(".csv"):
            cfg["output"]["csv_file"] = base_csv[:-4] + f"_{int(args.freq_mhz)}MHz.csv"
        if base_plot.endswith(".png"):
            cfg["output"]["plot_file"] = base_plot[:-4] + f"_{int(args.freq_mhz)}MHz.png"

    scanner = RFScanner(cfg)
    scanner.run()


if __name__ == "__main__":
    main()
