# 📡 RF Terrain Site Finder (VHF/UHF/SHF Contest Planning)


## Description

This tool  uses **terrain (DEM) data** to identify optimal locations for VHF/UHF/SHF contest operation.

There is no guarantee that this application will always generate correct data. This is a proof of concept (POC), and I assume no responsibility for its use.



Goal:
👉 find sites with **maximum RF visibility toward Western Europe**,  
taking into account real-world RF propagation effects:

- terrain horizon
- Fresnel clearance
- diffraction loss
- azimuth coverage

---

# Features


- Grid scan around a center location
- RF evaluation per azimuth
- Automatic site ranking
- Azimuth coverage analysis
- Output:
  - CSV file
  - heatmap
  - detailed per-site plots

---

#  Installation

```bash
sudo apt install python3 python3-pip gdal-bin
pip install numpy rasterio pyproj matplotlib
```

---

#  Usage

```bash
python3 site_scan_coverage_json.py
```

Different frequency:

```bash
python3 site_scan_coverage_json.py --freq-mhz 432
python3 site_scan_coverage_json.py --freq-mhz 1296
```

DEM data: https://portal.opentopography.org/
Download tif file
---

#  Configuration (`config.json`)

- `radius_km` – scan radius
- `grid_step_km` – grid resolution
- `freq_mhz` – operating frequency
- `azimuth range` – direction of interest

---

#  Results Interpretation

##  Metrics

### `score`
**:** combined RF score (0–1)

---

### `worst_horizon_deg`
**:**
- < 0° → excellent  
- 0–1° → acceptable  
- > 2° → problematic  

---

### `min_fc` (Fresnel clearance)
**:**
- > 10 m → excellent  
- 0–10 m → marginal  
- < 0 → obstructed  

---

### `diff` (diffraction loss)
**:**
- 0 dB → LOS  
- < 3 dB → acceptable  
- > 6 dB → severe loss  

---

### `usable`
**:** total usable azimuth width  

---

### `contig`
**:** largest contiguous usable sector  
- most important metric  

---

# Interpretation


###  Contest site
- contig ≥ 30°
- usable ≥ 40°
- low diffraction

 best for QSO count

---

### DX site
- high elevation
- excellent horizon
- narrow sector acceptable

---

###  Bad site
- contig < 10°
- blocked Fresnel
- high diffraction

---

#  Plots

## Cartesian plots
- horizon vs azimuth
- Fresnel clearance
- diffraction
- usable mask

## Polar plot
- radial = RF quality
- color = usable / blocked

---


# ToDo

- Multi-band scoring (144 / 432 / 1296 MHz)
- Real DX targets
- Link budget
- Tropospheric propagation
- Clutter modeling

---

# License

MIT
