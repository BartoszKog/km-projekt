"""Zbuduj geometrie tras KM wzdłuż torów OSM (per line_code).

Dla każdej linii z stations_lists.parquet:
  1. bierze współrzędne stacji ze schedule.parquet,
  2. snapuje je do najbliższego węzła grafu torów
     (railway_data_clean.parquet),
  3. łączy kolejne stacje shortest-path po torach,
  4. zapisuje geometrię do data/processed/km_line_geometries.parquet.

Uruchomienie (z katalogu głównego repo):
  uv run python scripts/build_km_line_geometries.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Snap stacji do węzła toru (metry, EPSG:2180).
MAX_SNAP_M = 800.0
# Zaokrąglenie węzłów grafu (metry) — łączy bliskie końce odcinków.
NODE_ROUND_M = 1.0


def _round_node(coord: tuple[float, ...]) -> tuple[float, float]:
    return (
        round(coord[0] / NODE_ROUND_M) * NODE_ROUND_M,
        round(coord[1] / NODE_ROUND_M) * NODE_ROUND_M,
    )


def load_inputs(
    processed: Path,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, list[str]]]:
    railway = gpd.read_parquet(processed / "railway_data_clean.parquet")
    if railway.crs is None:
        raise ValueError("railway_data_clean.parquet nie ma CRS")

    schedule = pd.read_parquet(processed / "schedule.parquet")
    stops = (
        schedule.groupby("stop_name", as_index=False)
        .agg(stop_lat=("stop_lat", "median"), stop_lon=("stop_lon", "median"))
    )
    # Odrzuć oczywiste błędy GTFS (np. BAŃSKA NIŻNA 0,0).
    stops = stops[
        (stops["stop_lat"].between(49.0, 51.5))
        & (stops["stop_lon"].between(18.5, 23.0))
    ].copy()

    stations_lists = pd.read_parquet(processed / "stations_lists.parquet")
    by_line = {
        row.line_code: list(row.station_sequence)
        for row in stations_lists.itertuples(index=False)
    }
    return railway, stops, by_line


def build_rail_graph(railway: gpd.GeoDataFrame) -> nx.Graph:
    """Graf nieskierowany: węzły = współrzędne, waga = długość odcinka [m]."""
    graph = nx.Graph()

    for geom in railway.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for part in parts:
            coords = [_round_node(c) for c in part.coords]
            cleaned: list[tuple[float, float]] = [coords[0]]
            for c in coords[1:]:
                if c != cleaned[-1]:
                    cleaned.append(c)
            for a, b in zip(cleaned, cleaned[1:]):
                weight = Point(a).distance(Point(b))
                if weight <= 0:
                    continue
                if graph.has_edge(a, b):
                    if weight < graph[a][b]["weight"]:
                        graph[a][b]["weight"] = weight
                else:
                    graph.add_edge(a, b, weight=weight)

    return graph


def snap_point_to_node(
    point: Point,
    nodes: list[tuple[float, float]],
    node_tree: STRtree,
    max_snap_m: float,
) -> tuple[tuple[float, float] | None, float]:
    idxs, dists = node_tree.query_nearest(point, return_distance=True)
    idx = int(idxs[0])
    dist = float(dists[0])
    if dist > max_snap_m:
        return None, dist
    return nodes[idx], dist


def route_between(
    graph: nx.Graph,
    a: tuple[float, float],
    b: tuple[float, float],
) -> LineString | None:
    if a == b:
        return None
    try:
        path = nx.shortest_path(graph, a, b, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(path) < 2:
        return None
    return LineString(path)


def build_line_geometry(
    station_names: list[str],
    stop_points: dict[str, Point],
    graph: nx.Graph,
    nodes: list[tuple[float, float]],
    node_tree: STRtree,
    max_snap_m: float,
) -> tuple[LineString | MultiLineString | None, dict]:
    """Zwraca geometrię trasy + statystyki snap/routingu."""
    stats = {
        "n_stations": len(station_names),
        "n_with_coords": 0,
        "n_snapped": 0,
        "n_hops": 0,
        "n_routed": 0,
        "n_failed_snap": 0,
        "n_failed_route": 0,
        "max_snap_m": 0.0,
        "missing_stops": [],
    }

    snapped: list[tuple[str, tuple[float, float]]] = []
    for name in station_names:
        pt = stop_points.get(name)
        if pt is None:
            stats["missing_stops"].append(name)
            continue
        stats["n_with_coords"] += 1
        node, dist = snap_point_to_node(pt, nodes, node_tree, max_snap_m)
        if dist < float("inf"):
            stats["max_snap_m"] = max(stats["max_snap_m"], dist)
        if node is None:
            stats["n_failed_snap"] += 1
            continue
        stats["n_snapped"] += 1
        snapped.append((name, node))

    segments: list[LineString] = []
    for (_n0, a), (_n1, b) in zip(snapped, snapped[1:]):
        stats["n_hops"] += 1
        geom = route_between(graph, a, b)
        if geom is None:
            stats["n_failed_route"] += 1
            continue
        stats["n_routed"] += 1
        segments.append(geom)

    if not segments:
        return None, stats

    merged = linemerge(unary_union(segments))
    if merged.is_empty:
        return None, stats
    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed",
        type=Path,
        default=PROCESSED,
        help="Katalog data/processed",
    )
    parser.add_argument(
        "--max-snap-m",
        type=float,
        default=MAX_SNAP_M,
        help="Maks. odległość snapu stacji do węzła toru [m]",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ścieżka wyjściowa parquet (domyślnie km_line_geometries.parquet)",
    )
    args = parser.parse_args()
    processed: Path = args.processed
    out_path = args.out or (processed / "km_line_geometries.parquet")

    print(f"Wczytuję dane z {processed} …")
    railway, stops, by_line = load_inputs(processed)
    print(
        f"  tory: {len(railway)} odcinków, stacje GTFS: {len(stops)}, "
        f"linie: {len(by_line)}"
    )

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(railway.crs)
    stop_points = {
        row.stop_name: row.geometry for row in stops_gdf.itertuples(index=False)
    }

    print("Buduję graf torów …")
    graph = build_rail_graph(railway)
    nodes = list(graph.nodes)
    node_tree = STRtree([Point(n) for n in nodes])
    print(
        f"  węzły={graph.number_of_nodes()}, "
        f"krawędzie={graph.number_of_edges()}"
    )

    rows: list[dict] = []
    for line_code, sequence in sorted(by_line.items()):
        print(f"Routing {line_code} ({len(sequence)} stacji) …")
        geom, stats = build_line_geometry(
            sequence,
            stop_points,
            graph,
            nodes,
            node_tree,
            max_snap_m=args.max_snap_m,
        )
        print(
            f"  snapped={stats['n_snapped']}/{stats['n_with_coords']} "
            f"routed={stats['n_routed']}/{stats['n_hops']} "
            f"fail_snap={stats['n_failed_snap']} "
            f"fail_route={stats['n_failed_route']} "
            f"max_snap={stats['max_snap_m']:.1f}m"
        )
        if stats["missing_stops"]:
            print(f"  brak coords: {stats['missing_stops']}")
        if geom is None:
            print(f"  UWAGA: brak geometrii dla {line_code}")
            continue
        length_km = float(geom.length) / 1000.0
        rows.append(
            {
                "line_code": line_code,
                "n_stations": stats["n_stations"],
                "n_snapped": stats["n_snapped"],
                "n_routed_hops": stats["n_routed"],
                "length_km": round(length_km, 3),
                "geometry": geom,
            }
        )

    if not rows:
        raise SystemExit("Nie udało się zbudować żadnej geometrii linii.")

    out = gpd.GeoDataFrame(rows, crs=railway.crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f"Zapisano {len(out)} linii → {out_path}")
    print(out.drop(columns="geometry").to_string(index=False))


if __name__ == "__main__":
    main()
