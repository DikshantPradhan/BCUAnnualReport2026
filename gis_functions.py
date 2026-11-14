import warnings
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import box, LineString, Point
from shapely.ops import linemerge, substring
import re

def segment_from_pts(pt1, pt2, name="segment", town="town", project="project", crs = "EPSG:26986"):
    segment = gpd.GeoDataFrame(
        {
            "name": [name],
            "MGIS_TOWN": [town],
            "PROJECT": [project]
        },
        geometry=[LineString([pt1, pt2])],
        crs=crs,
    )
    return(segment)

# def street_intersection(city, target_street, query_street):
#     name = city.STREETNAME.fillna("").str.upper()
#     tgt_st = city[name == target_street]
#     query_st = city[name == query_street]
    
#     # Merge the many Cambridge St segments into continuous lines
#     merged = linemerge(tgt_st.geometry.union_all())
#     # parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    
#     # Where each cross street meets Cambridge St
#     intersect_pt = query_st.geometry.union_all().intersection(merged)
#     # if len(intersect_pt) > 0:
#     #     intersect_pt = intersect_pt[0]
#     return(intersect_pt)

def street_intersection(city, target_street, query_street):
    name = city.STREETNAME.fillna("").str.upper()
    tgt_st = city[name == target_street]
    query_st = city[name == query_street]
    
    # Merge the segments into continuous lines
    merged = linemerge(tgt_st.geometry.union_all())
    # parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    intersect_pt = query_st.geometry.union_all().intersection(merged)
    return(intersect_pt)

def segment_from_intersections(city, start1, start2, end1, end2, 
                               name="name", project="project", crs = "EPSG:26986"):
    start_pt = street_intersection(city, start1, start2)
    end_pt = street_intersection(city, end1, end2)
    segment = segment_from_pts(start_pt, end_pt, name=name, town=city["MGIS_TOWN"].unique()[0], project=project, crs = crs)
    return(segment)

def street_segment_from_pts(city, target_street, point_1, point_2, 
                     from_st = "start", to_st = "end", project = "project", crs = "EPSG:26986"):
    name = city.STREETNAME.fillna("").str.upper()
    tgt_st = city[name == target_street]
    merged = linemerge(tgt_st.geometry.union_all())
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    
    part = next(p for p in parts if p.distance(point_1) < 1 and p.distance(point_2) < 1)
    d1, d2 = sorted([part.project(point_1), part.project(point_2)])
    
    # The segment as a GeoDataFrame (same CRS as the roads layer)
    segment = gpd.GeoDataFrame(
        {
            "STREETNAME": [target_street],
            "FROM_ST": [from_st],
            "TO_ST": [to_st],
            "MGIS_TOWN": [city["MGIS_TOWN"].unique()[0]],
            "PROJECT": [project],
            "LENGTH_M": [d2 - d1],
        },
        geometry=[substring(part, d1, d2)],
        crs=crs,
    )
    return(segment)

def street_segment_from_intersections(city, target_street, start_street, end_street, project="project"):
    start_pt = street_intersection(city, target_street, start_street)
    end_pt = street_intersection(city, target_street, end_street)
    segment = street_segment_from_pts(city, target_street, start_pt, end_pt, from_st=start_street, to_st=end_street, project=project)
    return(segment)

def segment_along_street(city, pt, target_street, length, dir_check_dis=25, tol=1,
                         name=None, project="project", crs="EPSG:26986", straight=True):
    # Segment of `length` meters along `target_street`, starting at the intersection point `pt`.

    #length > 0 goes north/east, length < 0 goes south/west.
    #Raises ValueError if the street isn't found or `pt` is more than `tol` meters from it.
    
    target_street = target_street.upper()
    streetname = city.STREETNAME.fillna("").str.upper()
    street = city[streetname == target_street]
    if street.empty:
        raise ValueError(f"No street named {target_street!r} in `city`")

    merged = linemerge(street.geometry.union_all())
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

    # the continuous piece of the street closest to the intersection
    part = min(parts, key=lambda p: p.distance(pt))

    # make sure the point is on (or touching) the target street
    gap = part.distance(pt)
    if gap > tol:
        raise ValueError(f"Point is {gap:.1f} m from {target_street}; it must be within {tol} m of the street")

    d0 = part.project(pt)

    # which way along the line is north/east? use the local direction at the intersection
    a = part.interpolate(max(d0 - dir_check_dis, 0))
    b = part.interpolate(min(d0 + dir_check_dis, part.length))
    dx, dy = b.x - a.x, b.y - a.y
    forward_is_ne = (dy >= 0) if abs(dy) >= abs(dx) else (dx >= 0)  # north for N-S streets, east for E-W

    # walk `length` meters in the requested direction, stopping at the end of the street
    step = length if forward_is_ne else -length
    d_end = min(max(d0 + step, 0), part.length)
    if abs(d_end - d0) < abs(length):
        warnings.warn(f"{target_street} ends before {abs(length):g} m; segment is {abs(d_end - d0):.0f} m long")
    end_pt = part.interpolate(d_end)
    if straight:
        return segment_from_pts(
            pt, end_pt,
            name=name or f"{target_street} {length:+g} m",
            town=city["MGIS_TOWN"].unique()[0],
            project=project,
            crs=crs,
        )
    else:
        return street_segment_from_pts(city, target_street, pt, end_pt, project=project, crs=crs)

def street_length(city, target_street, units="meters"): # alternate is miles
    # Total length of `target_street` in the layer's CRS units (meters for EPSG:26986).
    # Raises ValueError if the street isn't found.

    target_street = target_street.upper()
    streetname = city.STREETNAME.fillna("").str.upper()
    street = city[streetname == target_street]
    if street.empty:
        # raise ValueError(f"No street named {target_street!r} in `city`")
        return(-1)

    # union_all() dissolves duplicate/overlapping pieces so they aren't counted twice
    street_length = street.geometry.union_all().length
    if units=="miles":
        street_length = street_length / 1609.344
    return street_length

def street_from_coords(lat, lon, city, max_dist=100):
    # Nearest street address for a (lat, lon) point, e.g. "Congress Street".
    # `city` is a roads GeoDataFrame (columns STREETNAME, MGIS_TOWN) in any CRS.
    # Raises ValueError if the nearest street is more than `max_dist` meters away.

    # lat/lon -> the roads layer's CRS (meters for EPSG:26986)
    pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(city.crs)

    # nearest named road (sjoin_nearest returns ties, so sort and take the closest)
    named = city[city.STREETNAME.fillna("").str.strip() != ""]
    hit = gpd.sjoin_nearest(pt, named, distance_col="dist_m").sort_values("dist_m").iloc[0]

    if hit.dist_m > max_dist:
        # raise ValueError(f"Nearest street ({hit.STREETNAME}) is {hit.dist_m:.0f} m away; more than max_dist={max_dist} m")
        return "coord error"
    return hit.STREETNAME.title()

# def count_points_along_streets(city, streets, points, seg_length=100, dist=50,
#                                lat_col="latitude", long_col="longitude", crs_type="EPSG:4326"):
#     # Split each street in `streets` into sections of `seg_length` meters and count how many
#     # rows of `points` (a DataFrame with lat/lon columns) are within `dist` meters of each section.
#     # Returns a GeoDataFrame (CRS of `city`) with one row per segment and a `count` column.

#     pts = points.dropna(subset=[lat_col, long_col])
#     pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[long_col], pts[lat_col]), crs=crs_type).to_crs(city.crs)

#     rows = []
#     for street in streets:
#         street = street.upper()
#         match = city[city.STREETNAME.fillna("").str.upper() == street]
#         if match.empty:
#             raise ValueError(f"No street named {street!r} in `city`")

#         # one continuous line per connected piece of the street
#         geom = match.geometry.union_all()
#         merged = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
#         parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

#         for part_id, part in enumerate(parts):
#             for i, start in enumerate(np.arange(0, part.length, seg_length)):
#                 end = min(start + seg_length, part.length)   # last section of a piece may be shorter
#                 rows.append({"STREETNAME": street, "part": part_id, "segment": i,
#                              "start_m": start, "end_m": end, "length_m": end - start,
#                              "geometry": substring(part, start, end)})

#     segs = gpd.GeoDataFrame(rows, geometry="geometry", crs=city.crs)

#     # count points within `dist` of each segment (buffer, then spatial join)
#     buf = segs[["geometry"]].copy()
#     buf["geometry"] = segs.buffer(dist)
#     joined = gpd.sjoin(buf, pts, how="left", predicate="contains")
#     segs["count"] = joined["index_right"].notna().groupby(level=0).sum().reindex(segs.index).astype(int)
#     return segs

def split_streets(city, streets, seg_length=100, frame_length=-1):
    # Merge all pieces of each street in `streets` into one ordered route, then cut it into
    # sections of `seg_length` meters. Returns a GeoDataFrame (CRS of `city`), one row per segment.
    if frame_length < 0:
        frame_length = seg_length

    rows = []
    for street in streets:
        street = street.upper()
        match = city[city.STREETNAME.fillna("").str.upper() == street]
        if match.empty:
            continue
            # raise ValueError(f"No street named {street!r} in `city`")

        geom = match.geometry.union_all()
        merged = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
        parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        parts = [p for p in parts if p.length > 0]

        # point every piece the same way (west->east for E-W streets, south->north for N-S),
        # then order the pieces along the street
        minx, miny, maxx, maxy = merged.bounds
        axis = 0 if (maxx - minx) >= (maxy - miny) else 1
        parts = [LineString(list(p.coords)[::-1]) if p.coords[0][axis] > p.coords[-1][axis] else p for p in parts]
        parts.sort(key=lambda p: p.coords[0][axis])

        # treat the pieces as one long street: piece k starts at offsets[k] meters
        offsets = np.concatenate([[0], np.cumsum([p.length for p in parts])])
        total = offsets[-1]

        for i, start in enumerate(np.arange(0, total, frame_length)):
            end = min(start + seg_length, total)
            # if i > 0 and end - start < seg_length - 1e-9: break   # stop once a window is shorter than seg_length
            pieces = []
            for p, off in zip(parts, offsets):
                a, b = max(start, off) - off, min(end, off + p.length) - off
                if b - a > 1e-9:
                    pieces.append(substring(p, a, b))
            geom_i = pieces[0] if len(pieces) == 1 else linemerge(pieces)   # MultiLineString if it spans a gap
            rows.append({"STREETNAME": street, "segment": i, "start_m": start, "end_m": end,
                         "length_m": end - start,"geometry": geom_i})

    df = gpd.GeoDataFrame(rows, geometry="geometry", crs=city.crs)
    df['segment_id'] = df['STREETNAME'] + df['segment'].astype(str)
    return df

def count_points_along_segments(segments, points, dist=5, lat_col="latitude", long_col="longitude", crs_type="EPSG:4326"):
    pts = points.dropna(subset=[lat_col, long_col])
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts[long_col], pts[lat_col]), crs=crs_type).to_crs(segments.crs)

    # count points within `dist` of each segment (buffer, then spatial join)
    buf = segments[["geometry"]].copy()
    buf["geometry"] = segments.buffer(dist)
    joined = gpd.sjoin(buf, pts, how="left", predicate="contains")
    segments["count"] = joined["index_right"].notna().groupby(level=0).sum().reindex(segments.index).astype(int)
    return segments

def plot_segment_counts(segments, count_col="count", 
                        cmap=LinearSegmentedColormap.from_list("blue_red", ["#2b6cb0", "#bdbdbd", "#c62828"]), 
                        background=None,
                        linewidth=4, title=None, ax=None, figsize=(10, 10), **kwargs):
    # Plot the output of count_points_along_streets, colored by count (blue = fewer, red = more).
    # `background` is an optional GeoDataFrame (e.g. the city's roads) drawn in light gray underneath.
    # Extra keyword arguments (e.g. vmin=, vmax=) go to GeoDataFrame.plot.

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if background is not None:
        background.plot(ax=ax, color="#d9d5cc", linewidth=0.6, zorder=1)

    segments.plot(ax=ax, column=count_col, cmap=cmap, linewidth=linewidth, zorder=2,
                  legend=True, legend_kwds={"label": "Points within distance of segment", "shrink": 0.6},
                  **kwargs)

    minx, miny, maxx, maxy = segments.total_bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax

def fix_street_abbrevs(st_list):
    fixed_streets = []
    for st in st_list:  
        # fx_st = re.sub(s1, s2)
        st = re.sub(' ALY$', " ALLEY", st)
        st = re.sub(' AVE$', " AVENUE", st)
        st = re.sub(' BLVD$', " BOULEVARD", st)
        st = re.sub(' CIR$', " CIRCLE", st)
        st = re.sub(' CT$', " COURT", st)
        st = re.sub(' DR$', " DRIVE", st)
        st = re.sub(' HWY$', " HIGHWAY", st)
        st = re.sub(' LN$', " LANE", st)
        st = re.sub(' PKWY$', " PARKWAY", st)
        st = re.sub(' PL$', " PLACE", st)
        st = re.sub(' PLZ$', " PLAZA", st)
        st = re.sub(' RD$', " ROAD", st)
        st = re.sub(' SQ$', " SQUARE", st)
        st = re.sub(' ST$', " STREET", st)
        st = re.sub(' TER$', " TERRACE", st)
        st = re.sub(' WHRF$', " WHARF", st)
        st = re.sub(' NULL$', "", st)
        st = re.sub(' BRG$', " BRIDGE", st)
        st = re.sub('\\.', "", st)

        ## specific changes
        # st = re.sub('CENTER PLAZA$', "CAMBRIDGE STREET", st)
        st = re.sub('THIRTEENTH STREET$', "13TH STREET", st)
        st = re.sub('SUDBURY STREET$', "NEW SUDBURY STREET", st)
        st = st.strip()
        fixed_streets.append(st) 
    return(fixed_streets)
