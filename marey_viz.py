"""Wykres Mareya KM — drop-in kontrolka (dane → UI / widoki).

Moduł do importu z dowolnego notebooka marimo lub skryptu. Nie zależy od
komórek `marey_chart.py`; ten notebook jest tylko przykładem wiringu UI.

Quick start (parametry znane z góry)
------------------------------------
```python
from datetime import date
from marey_viz import load_marey_data, marey_explorer

data = load_marey_data()  # data/processed/
marey_explorer(
    data,
    selected_date=date(2026, 10, 23),
    line_code="SKA3P",
    trip_id="<trip_id z GTFS>",
    margin_minutes=60,
)
```

Quick start (2 komórki marimo: data + formularz)
-----------------------------------------------
```python
import marimo as mo
from marey_viz import load_marey_data, marey_explorer_form, marey_from_form

data = load_marey_data()
date_picker = mo.ui.date(
    value=data.available_dates[0],
    start=data.available_dates[0],
    stop=data.available_dates[-1],
    label="Data",
)
date_picker
```

```python
form = marey_explorer_form(data, selected_date=date_picker.value)
form
```

```python
marey_from_form(data, form.value, selected_date=date_picker.value)
```

Kontrakt danych
---------------
``schedule`` — m.in. ``trip_id``, ``service_id``, ``line_code``,
``stop_name``, ``stop_sequence``, ``arrival_seconds``, ``departure_seconds``,
``trip_short_name``, ``stop_lat``, ``stop_lon``.

``calendar_dates`` — ``service_id``, ``date`` (``datetime.date``).

``stations_by_line`` — ``dict[line_code, list[stop_name]]`` (kolejność osi).

``line_geoms`` — GeoDataFrame z ``line_code``, ``length_km``, ``geometry``
(wynik ``scripts/build_km_line_geometries.py``).

``railway`` — tory OSM (tło mapy), dowolne LineString w metrycznym CRS.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import altair as alt
import geopandas as gpd
import marimo as mo
import pandas as pd

# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

DEFAULT_KOLORY_LINII_KML: dict[str, str] = {
    "SKA1": "#009FE3",
    "SKA2": "#DF2D84",
    "SKA3": "#F5B620",
    "SKA3P": "#F5B620",
    "K5": "#F29EC4",
    "K5P": "#F29EC4",
    "K7": "#008232",
    "K7P": "#008232",
    "K71": "#28235B",
    "K71P": "#28235B",
    "K63": "#4464AC",
    "TLO_PKP": "#D0D5DD",
    "AKTYWNA_SELEKCJA": "#FF2D55",
}


class MareyData(NamedTuple):
    """Spakowane dane wejściowe dla kontrolki Mareya."""

    schedule: pd.DataFrame
    calendar_dates: pd.DataFrame
    stations_by_line: dict[str, list[str]]
    railway: gpd.GeoDataFrame
    line_geoms: gpd.GeoDataFrame
    available_dates: list[date]


class MareyResult(NamedTuple):
    """Wynik pipeline'u przed renderem UI."""

    related: pd.DataFrame
    station_list: list[str]
    selected_trip_id: str
    selected_line: str
    n_same: int
    n_opp: int


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def resolve_processed_dir(processed: Path | str | None = None) -> Path:
    """Znajdź katalog ``data/processed`` (cwd albo rodzic)."""
    if processed is not None:
        return Path(processed)
    for candidate in (Path("data/processed"), Path("..") / "data" / "processed"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Nie znaleziono data/processed — podaj processed=Path(...)."
    )


def load_marey_data(processed: Path | str | None = None) -> MareyData:
    """Wczytaj parquet-y potrzebne do wykresu / mapy / tabeli.

    Parameters
    ----------
    processed
        Katalog z ``schedule.parquet``, ``calendar_dates.parquet``,
        ``stations_lists.parquet``, ``railway_data_clean.parquet``,
        ``km_line_geometries.parquet``. Domyślnie auto-detect.

    Returns
    -------
    MareyData
    """
    root = resolve_processed_dir(processed)
    schedule = pd.read_parquet(root / "schedule.parquet")
    calendar_dates = pd.read_parquet(root / "calendar_dates.parquet").copy()
    calendar_dates["date"] = pd.to_datetime(calendar_dates["date"]).dt.date

    stations_lists = pd.read_parquet(root / "stations_lists.parquet")
    stations_by_line = {
        row.line_code: list(row.station_sequence)
        for row in stations_lists.itertuples(index=False)
    }

    railway = gpd.read_parquet(root / "railway_data_clean.parquet")
    line_geoms = gpd.read_parquet(root / "km_line_geometries.parquet")
    available_dates = sorted(calendar_dates["date"].unique())

    return MareyData(
        schedule=schedule,
        calendar_dates=calendar_dates,
        stations_by_line=stations_by_line,
        railway=railway,
        line_geoms=line_geoms,
        available_dates=available_dates,
    )


# ---------------------------------------------------------------------------
# Czas / etykiety
# ---------------------------------------------------------------------------


def seconds_to_label(seconds: int) -> str:
    """HH:MM z uwzględnieniem czasu > 24h (GTFS)."""
    day = seconds // 86400
    rem = seconds % 86400
    h, rem = divmod(rem, 3600)
    m = rem // 60
    label = f"{h:02d}:{m:02d}"
    if day:
        label = f"{label}+{day}d"
    return label


def trip_label(g: pd.DataFrame) -> str:
    """Etykieta kursu do dropdowna: godzina · numer · (A → B)."""
    g = g.sort_values("stop_sequence")
    dep = seconds_to_label(int(g["departure_seconds"].iloc[0]))
    name = g["trip_short_name"].iloc[0]
    origin = g["stop_name"].iloc[0]
    dest = g["stop_name"].iloc[-1]
    return f"{dep}  {name}  ({origin} → {dest})"


# ---------------------------------------------------------------------------
# Oś stacji / kierunek
# ---------------------------------------------------------------------------


def station_name_index(station_list: list[str]) -> dict[str, list[int]]:
    """``stop_name`` → wszystkie indeksy na osi linii (powtórzenia)."""
    index: dict[str, list[int]] = {}
    for i, name in enumerate(station_list):
        index.setdefault(name, []).append(i)
    return index


def map_trip_to_axis(
    stop_names: list[str],
    name_index: dict[str, list[int]],
    prefer_increasing: bool,
) -> list[int | None]:
    """Przypisz przystanki kursu do indeksów osi (greedy, z kierunkiem)."""
    used: set[int] = set()
    result: list[int | None] = []
    prev: int | None = None

    for name in stop_names:
        candidates = [i for i in name_index.get(name, []) if i not in used]
        if not candidates:
            result.append(None)
            continue
        if prev is None:
            pick = candidates[0] if prefer_increasing else candidates[-1]
        elif prefer_increasing:
            ahead = [i for i in candidates if i >= prev]
            pick = min(ahead) if ahead else min(candidates, key=lambda i: abs(i - prev))
        else:
            behind = [i for i in candidates if i <= prev]
            pick = max(behind) if behind else min(candidates, key=lambda i: abs(i - prev))
        used.add(pick)
        result.append(pick)
        prev = pick
    return result


def trip_prefer_increasing(
    stop_names: list[str], station_list: list[str]
) -> bool:
    """Czy kurs idzie zgodnie z kolejnością ``station_list``."""
    name_index = station_name_index(station_list)
    first = next((name_index[n][0] for n in stop_names if n in name_index), None)
    last = next(
        (name_index[n][-1] for n in reversed(stop_names) if n in name_index), None
    )
    if first is None or last is None:
        return True
    return last >= first


def trip_direction_sign(station_indices: list[int]) -> int:
    """+1 zgodnie z rosnącym indeksem, -1 przeciwnie, 0 niejasne."""
    if len(station_indices) < 2:
        return 0
    delta = station_indices[-1] - station_indices[0]
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def trip_time_span(g: pd.DataFrame) -> tuple[int, int]:
    """Min przyjazdu / max odjazdu w ramce przystanków."""
    t0 = int(min(g["arrival_seconds"].min(), g["departure_seconds"].min()))
    t1 = int(max(g["arrival_seconds"].max(), g["departure_seconds"].max()))
    return t0, t1


# ---------------------------------------------------------------------------
# Filtrowanie kursów
# ---------------------------------------------------------------------------


def day_schedule(
    schedule: pd.DataFrame, calendar_dates: pd.DataFrame, selected_date: date
) -> pd.DataFrame:
    """Rozkład na wybrany dzień kalendarza GTFS."""
    services = calendar_dates.loc[
        calendar_dates["date"] == selected_date, "service_id"
    ]
    return schedule[schedule["service_id"].isin(services)].copy()


def related_trips(
    day: pd.DataFrame,
    selected_trip_id: str,
    station_list: list[str],
    margin_seconds: int = 3600,
    visible_indices: set[int] | None = None,
) -> pd.DataFrame:
    """Kursy dzielące ≥2 stacje z widocznym odcinkiem, w oknie czasowym.

    Role: ``selected`` / ``same`` / ``opposite`` względem kierunku
    wybranego kursu. Okno czasowe liczone tylko na ``visible_indices``.
    """
    sel = day[day["trip_id"] == selected_trip_id].sort_values("stop_sequence")
    name_index = station_name_index(station_list)
    visible = (
        set(visible_indices)
        if visible_indices is not None
        else set(range(len(station_list)))
    )

    sel_names = sel["stop_name"].tolist()
    sel_increasing = trip_prefer_increasing(sel_names, station_list)
    sel_axis = map_trip_to_axis(
        sel_names, name_index, prefer_increasing=sel_increasing
    )
    sel_mapped = sel.copy()
    sel_mapped["station_idx"] = sel_axis
    sel_visible = sel_mapped[sel_mapped["station_idx"].isin(visible)].copy()
    if len(sel_visible) < 2:
        return pd.DataFrame()

    sel_visible["station_idx"] = sel_visible["station_idx"].astype(int)
    t0, t1 = trip_time_span(sel_visible)
    selected_sign = trip_direction_sign(
        sel_visible.sort_values("departure_seconds")["station_idx"].tolist()
    )

    rows: list[pd.DataFrame] = []
    for trip_id, g in day.groupby("trip_id", sort=False):
        g = g.sort_values("stop_sequence")
        names = g["stop_name"].tolist()
        increasing = trip_prefer_increasing(names, station_list)
        axis_idx = map_trip_to_axis(
            names, name_index, prefer_increasing=increasing
        )
        out = g.copy()
        out["station_idx"] = axis_idx
        out = out[out["station_idx"].isin(visible)].copy()
        if len(out) < 2:
            continue
        out["station_idx"] = out["station_idx"].astype(int)
        out["axis_station"] = out["station_idx"].map(lambda i: station_list[i])

        if trip_id != selected_trip_id:
            gt0, gt1 = trip_time_span(out)
            if gt1 < t0 - margin_seconds or gt0 > t1 + margin_seconds:
                continue
            idxs = out.sort_values("departure_seconds")["station_idx"].tolist()
            cand_sign = trip_direction_sign(idxs)
            if selected_sign != 0 and cand_sign != 0 and cand_sign != selected_sign:
                role = "opposite"
            else:
                role = "same"
        else:
            role = "selected"

        out["role"] = role
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def trip_corridor_indices(
    day: pd.DataFrame,
    trip_id: str,
    station_list: list[str],
) -> list[int]:
    """Indeksy stacji linii w kolejności jazdy kursu (z pośrednimi)."""
    g = day[day["trip_id"] == trip_id].sort_values("stop_sequence")
    names = g["stop_name"].tolist()
    name_index = station_name_index(station_list)
    increasing = trip_prefer_increasing(names, station_list)
    mapped = [
        i
        for i in map_trip_to_axis(names, name_index, prefer_increasing=increasing)
        if i is not None
    ]
    if len(mapped) < 2:
        return mapped
    start, end = mapped[0], mapped[-1]
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, end - 1, -1))


def corridor_dropdown_options(
    corridor_indices: list[int], station_list: list[str]
) -> dict[str, int]:
    """Lokalny indeks wzdłuż kursu → etykieta; wartość = lokalny indeks."""
    return {
        f"{local:02d} · {station_list[global_idx]}": local
        for local, global_idx in enumerate(corridor_indices)
    }


def slice_related_to_corridor(
    related: pd.DataFrame,
    corridor_indices: list[int],
    station_list: list[str],
    local_from: int,
    local_to: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Przytnij do odcinka wzdłuż korytarza kursu (kolejność = kierunek jazdy)."""
    lo, hi = sorted((int(local_from), int(local_to)))
    visible_global = corridor_indices[lo : hi + 1]
    visible = [station_list[i] for i in visible_global]
    global_to_local = {g: i for i, g in enumerate(visible_global)}
    clipped = related[related["station_idx"].isin(global_to_local)].copy()
    clipped["station_idx"] = clipped["station_idx"].map(global_to_local).astype(int)
    clipped["axis_station"] = clipped["station_idx"].map(lambda i: visible[i])
    counts = clipped.groupby("trip_id").size()
    keep_ids = set(counts[counts >= 1].index)
    clipped = clipped[clipped["trip_id"].isin(keep_ids)]
    return clipped, visible


# ---------------------------------------------------------------------------
# Opcje do UI
# ---------------------------------------------------------------------------


def marey_line_options(data: MareyData, selected_date: date) -> list[str]:
    """Kody linii obecne w rozkładzie wybranego dnia."""
    day = day_schedule(data.schedule, data.calendar_dates, selected_date)
    return sorted(day["line_code"].dropna().unique().tolist())


def marey_trip_options(
    data: MareyData, selected_date: date, line_code: str
) -> dict[str, str]:
    """Mapa etykieta kursu → ``trip_id`` dla linii i dnia."""
    day = day_schedule(data.schedule, data.calendar_dates, selected_date)
    line_day = day[day["line_code"] == line_code]
    trip_groups = {tid: g for tid, g in line_day.groupby("trip_id", sort=False)}
    trip_ids_sorted = sorted(
        trip_groups,
        key=lambda tid: int(trip_groups[tid]["departure_seconds"].min()),
    )
    return {trip_label(trip_groups[tid]): tid for tid in trip_ids_sorted}


# ---------------------------------------------------------------------------
# Wykres / mapa / tabela
# ---------------------------------------------------------------------------


def marey_path_points(related: pd.DataFrame) -> pd.DataFrame:
    """Punkty ścieżki: przyjazd → (postój) odjazd → następny przyjazd."""
    base = pd.Timestamp("2000-01-01")
    points: list[dict] = []

    for trip_id, g in related.groupby("trip_id", sort=False):
        g = g.sort_values("stop_sequence")
        role = g["role"].iloc[0]
        line_code = g["line_code"].iloc[0]
        short = g["trip_short_name"].iloc[0]
        order = 0
        for r in g.itertuples(index=False):
            arr = int(r.arrival_seconds)
            dep = int(r.departure_seconds)
            y = int(r.station_idx)
            station = r.axis_station
            common = dict(
                trip_id=trip_id,
                role=role,
                line_code=line_code,
                trip_short_name=short,
                station_idx=y,
                station=station,
                label=f"{short} · {line_code}",
            )
            points.append(
                {
                    **common,
                    "seconds": arr,
                    "time": base + pd.Timedelta(seconds=arr),
                    "event": "przyjazd",
                    "order": order,
                }
            )
            order += 1
            if dep != arr:
                points.append(
                    {
                        **common,
                        "seconds": dep,
                        "time": base + pd.Timedelta(seconds=dep),
                        "event": "odjazd (postój)",
                        "order": order,
                    }
                )
                order += 1
            else:
                points[-1]["event"] = "przyjazd/odjazd"

    return pd.DataFrame(points)


def station_axis_label_expr(station_list: list[str]) -> str:
    """Mapowanie indeks → nazwa stacji w Vega expression."""
    items = []
    for i, name in enumerate(station_list):
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        items.append(f"'{i}':'{safe}'")
    return "{" + ",".join(items) + "}['' + datum.value]"


def marey_chart(
    related: pd.DataFrame,
    station_list: list[str],
    kolory: dict[str, str],
    name: str,
) -> alt.Chart:
    """Altair: wykres Mareya dla ``related`` na osi ``station_list``."""
    path = marey_path_points(related)
    role_labels = {
        "selected": "wybrany",
        "same": "ten sam kierunek",
        "opposite": "przeciwny kierunek",
    }
    aktywna = kolory["AKTYWNA_SELEKCJA"]
    path = path.assign(
        rola=path["role"].map(role_labels),
        legend_key=path.apply(
            lambda r: (
                "wybrany kurs" if r["role"] == "selected" else r["line_code"]
            ),
            axis=1,
        ),
    )

    line_codes_present = sorted(
        c for c in path.loc[path["role"] != "selected", "line_code"].unique()
    )
    legend_domain = ["wybrany kurs", *line_codes_present]
    legend_range = [
        aktywna,
        *[kolory.get(c, "#666666") for c in line_codes_present],
    ]
    color_scale = alt.Scale(domain=legend_domain, range=legend_range)

    dash_scale = alt.Scale(
        domain=["wybrany", "ten sam kierunek", "przeciwny kierunek"],
        range=[[1, 0], [1, 0], [6, 4]],
    )
    size_scale = alt.Scale(
        domain=["wybrany", "ten sam kierunek", "przeciwny kierunek"],
        range=[3.5, 1.6, 1.6],
    )

    y_enc = alt.Y(
        "station_idx:Q",
        title=None,
        scale=alt.Scale(domain=[len(station_list) - 0.5, -0.5], nice=False),
        axis=alt.Axis(
            values=list(range(len(station_list))),
            labelExpr=station_axis_label_expr(station_list),
            labelLimit=220,
            labelFontSize=10,
            tickMinStep=1,
            grid=True,
            gridOpacity=0.25,
            domain=False,
            ticks=False,
        ),
    )

    tooltip = [
        alt.Tooltip("label:N", title="Kurs"),
        alt.Tooltip("station:N", title="Stacja"),
        alt.Tooltip("time:T", title="Czas", format="%H:%M:%S"),
        alt.Tooltip("event:N", title="Zdarzenie"),
        alt.Tooltip("rola:N", title="Rola"),
    ]

    lines = (
        alt.Chart(path)
        .mark_line()
        .encode(
            x=alt.X(
                "time:T",
                title="Godzina",
                axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=12),
            ),
            y=y_enc,
            color=alt.Color(
                "legend_key:N",
                scale=color_scale,
                legend=alt.Legend(title="Linia"),
            ),
            strokeDash=alt.StrokeDash(
                "rola:N",
                scale=dash_scale,
                legend=alt.Legend(title="Styl", symbolStrokeWidth=2),
            ),
            strokeWidth=alt.StrokeWidth("rola:N", scale=size_scale, legend=None),
            detail="trip_id:N",
            order="order:Q",
            tooltip=tooltip,
            opacity=alt.condition(
                "datum.role === 'selected'",
                alt.value(1.0),
                alt.value(0.75),
            ),
        )
    )

    points = (
        alt.Chart(path)
        .mark_circle()
        .encode(
            x="time:T",
            y=y_enc,
            color=alt.Color("legend_key:N", scale=color_scale, legend=None),
            size=alt.condition(
                "datum.role === 'selected'",
                alt.value(60),
                alt.value(28),
            ),
            opacity=alt.condition(
                "datum.role === 'selected'",
                alt.value(1.0),
                alt.value(0.75),
            ),
            tooltip=tooltip,
        )
    )

    height = max(420, 26 * len(station_list))
    return (
        (lines + points)
        .properties(
            width="container",
            height=height,
            title=f"Ruch kolejowy w pobliżu linii: {name}",
        )
        .configure_view(strokeWidth=0)
    )


def marey_routes_map(
    line_geoms: gpd.GeoDataFrame,
    railway: gpd.GeoDataFrame,
    line_codes: list[str],
    selected_line: str,
    kolory: dict[str, str],
    station_points: pd.DataFrame | None = None,
):
    """Mapa Folium: trasy KM + tło torów OSM + legenda.

    Returns
    -------
    folium.Map | None
        ``None`` gdy brak geometrii dla podanych ``line_codes``.
    """
    import json

    import folium

    routes = line_geoms[line_geoms["line_code"].isin(line_codes)].to_crs(epsg=4326)
    if routes.empty:
        return None

    rail = railway.to_crs(epsg=4326)
    bounds = routes.total_bounds
    minx, miny, maxx, maxy = bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    m = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")

    bg = rail.cx[minx:maxx, miny:maxy]
    if not bg.empty:
        folium.GeoJson(
            json.loads(bg[["geometry"]].to_json()),
            name="Tory OSM (tło)",
            style_function=lambda _feature: {
                "color": kolory.get("TLO_PKP", "#D0D5DD"),
                "weight": 1,
                "opacity": 0.55,
            },
        ).add_to(m)

    ordered = sorted(c for c in line_codes if c != selected_line)
    if selected_line in set(routes["line_code"]):
        ordered.append(selected_line)

    for code in ordered:
        subset = routes[routes["line_code"] == code]
        if subset.empty:
            continue
        color = kolory.get(code, "#666666")
        weight = 6 if code == selected_line else 3.5
        opacity = 0.95 if code == selected_line else 0.8
        geojson = json.loads(
            subset[["geometry", "line_code", "length_km"]].to_json()
        )
        group = folium.FeatureGroup(name=code, show=True)
        folium.GeoJson(
            geojson,
            style_function=lambda _feature, c=color, w=weight, o=opacity: {
                "color": c,
                "weight": w,
                "opacity": o,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["line_code", "length_km"],
                aliases=["Linia:", "Długość [km]:"],
                localize=True,
            ),
        ).add_to(group)
        group.add_to(m)

    if station_points is not None and not station_points.empty:
        stops_group = folium.FeatureGroup(name="Stacje (zakres)", show=True)
        for row in station_points.itertuples(index=False):
            folium.CircleMarker(
                location=[row.stop_lat, row.stop_lon],
                radius=4,
                color="#222222",
                fill=True,
                fill_color="#ffffff",
                fill_opacity=0.95,
                weight=1,
                tooltip=row.stop_name,
            ).add_to(stops_group)
        stops_group.add_to(m)

    legend_items: list[tuple[str, str, str]] = []
    for code in ordered:
        color = kolory.get(code, "#666666")
        label = f"{code} (wybrana)" if code == selected_line else code
        swatch = (
            f'<span style="display:inline-block;width:22px;height:0;'
            f'border-top:4px solid {color};vertical-align:middle;'
            f'margin-right:8px;"></span>'
        )
        legend_items.append((code, swatch, label))

    tlo = kolory.get("TLO_PKP", "#D0D5DD")
    legend_items.append(
        (
            "_tlo",
            f'<span style="display:inline-block;width:22px;height:0;'
            f'border-top:2px solid {tlo};vertical-align:middle;'
            f'margin-right:8px;"></span>',
            "Tory OSM (tło)",
        )
    )
    if station_points is not None and not station_points.empty:
        legend_items.append(
            (
                "_stacje",
                '<span style="display:inline-block;width:10px;height:10px;'
                "border-radius:50%;border:1.5px solid #222;background:#fff;"
                'vertical-align:middle;margin:0 11px 0 6px;"></span>',
                "Stacje (zakres)",
            )
        )

    rows_html = "".join(
        f'<div style="margin:4px 0;font-size:13px;line-height:1.3;">'
        f"{swatch}{label}</div>"
        for _key, swatch, label in legend_items
    )
    legend_html = f"""
    <div style="
        position: fixed; bottom: 28px; left: 28px; z-index: 9999;
        background: rgba(255,255,255,0.94); padding: 10px 14px;
        border: 1px solid #c8cdd5; border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        font-family: system-ui, sans-serif; color: #1a1a1a; min-width: 140px;
    ">
      <div style="font-weight:600;font-size:13px;margin-bottom:6px;">Legenda</div>
      {rows_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(24, 24))
    return m


def marey_overview_table(related: pd.DataFrame) -> pd.DataFrame:
    """Tabela kursów w oknie (start / koniec / rola / stacje na korytarzu)."""
    return (
        related.groupby(
            ["trip_id", "line_code", "trip_short_name", "role"], as_index=False
        )
        .agg(
            first_dep=("departure_seconds", "min"),
            last_arr=("arrival_seconds", "max"),
            n_stops=("stop_id", "nunique"),
        )
        .assign(
            first_dep=lambda d: d["first_dep"].map(seconds_to_label),
            last_arr=lambda d: d["last_arr"].map(seconds_to_label),
            role=lambda d: d["role"].map(
                {
                    "selected": "wybrany",
                    "same": "ten sam kierunek",
                    "opposite": "przeciwny kierunek",
                }
            ),
        )
        .sort_values(["role", "first_dep"])
        .rename(
            columns={
                "line_code": "linia",
                "trip_short_name": "kurs",
                "role": "rola",
                "first_dep": "start",
                "last_arr": "koniec",
                "n_stops": "stacje na korytarzu",
            }
        )
        .drop(columns=["trip_id"])
    )


def marey_summary(result: MareyResult) -> Any:
    """Krótki opis markdown wybranego kursu i zakresu."""
    sel_meta = result.related[
        result.related["trip_id"] == result.selected_trip_id
    ].iloc[0]
    return mo.md(
        f"""
        **Wybrany kurs:** `{sel_meta["trip_short_name"]}` · linia **{sel_meta["line_code"]}**  
        Zakres wzdłuż kursu: **{result.station_list[0]}** → **{result.station_list[-1]}**
        ({len(result.station_list)} stacji na osi Y)  
        Okno czasowe liczone tylko na widocznych stacjach.  
        Poziomy odcinek = postój na stacji.
        """
    )


def marey_panel(
    related: pd.DataFrame,
    station_list: list[str],
    line_geoms: gpd.GeoDataFrame,
    railway: gpd.GeoDataFrame,
    selected_line: str,
    *,
    kolory: dict[str, str] | None = None,
    chart_title: str = "",
) -> Any:
    """Zakładki: wykres Mareya · mapa tras · tabela kursów.

    Parameters
    ----------
    related
        Wynik ``related_trips`` / ``marey_compute`` (z kolumną ``role``).
    station_list
        Oś Y (kolejność = kierunek jazdy na wykresie).
    line_geoms, railway
        Geometrie tras KM i tło torów.
    selected_line
        Kod linii do pogrubienia na mapie.
    kolory
        Paleta; domyślnie ``DEFAULT_KOLORY_LINII_KML``.
    chart_title
        Suffix tytułu wykresu Altair.
    """
    colors = kolory or DEFAULT_KOLORY_LINII_KML
    title = chart_title or selected_line

    chart = mo.ui.altair_chart(
        marey_chart(related, station_list, colors, title)
    )

    visible_lines = sorted(related["line_code"].dropna().unique().tolist())
    axis_stops = (
        related.loc[
            related["stop_name"].isin(station_list),
            ["stop_name", "stop_lat", "stop_lon"],
        ]
        .drop_duplicates(subset="stop_name")
        .copy()
    )
    order = {name: i for i, name in enumerate(station_list)}
    axis_stops["_ord"] = axis_stops["stop_name"].map(order)
    axis_stops = axis_stops.sort_values("_ord").drop(columns="_ord")

    routes_map = marey_routes_map(
        line_geoms,
        railway,
        visible_lines,
        selected_line=selected_line,
        kolory=colors,
        station_points=axis_stops,
    )
    map_view: Any = (
        mo.Html(routes_map._repr_html_())
        if routes_map is not None
        else mo.md(
            "Brak geometrii tras dla linii widocznych na wykresie "
            f"({', '.join(visible_lines)}). "
            "Uruchom `uv run python scripts/build_km_line_geometries.py`."
        )
    )

    return mo.ui.tabs(
        {
            "Wykres ruchu kolejowego": chart,
            "Mapa tras": map_view,
            "Tabela kursów": marey_overview_table(related),
        }
    )


# ---------------------------------------------------------------------------
# Pipeline + drop-in
# ---------------------------------------------------------------------------


def marey_compute(
    data: MareyData,
    *,
    selected_date: date,
    line_code: str,
    trip_id: str,
    local_from: int = 0,
    local_to: int | None = None,
    margin_minutes: int = 60,
) -> MareyResult:
    """Policz ``related`` + oś stacji dla zadanego wyboru (bez UI).

    Raises
    ------
    ValueError
        Gdy brak kursów, niemapowalny trip albo pusty wynik.
    """
    if line_code not in data.stations_by_line:
        raise ValueError(f"Nieznany line_code: {line_code!r}")

    day = day_schedule(data.schedule, data.calendar_dates, selected_date)
    full_station_list = data.stations_by_line[line_code]
    corridor_indices = trip_corridor_indices(day, trip_id, full_station_list)
    if len(corridor_indices) < 2:
        raise ValueError("Kurs nie da się zmapować na stacje linii.")

    lo = int(local_from)
    hi = int(len(corridor_indices) - 1 if local_to is None else local_to)
    lo, hi = sorted((lo, hi))
    lo = max(0, min(lo, len(corridor_indices) - 1))
    hi = max(0, min(hi, len(corridor_indices) - 1))

    margin_seconds = int(margin_minutes) * 60
    visible_global = set(corridor_indices[lo : hi + 1])
    related_full = related_trips(
        day,
        trip_id,
        full_station_list,
        margin_seconds=margin_seconds,
        visible_indices=visible_global,
    )
    if related_full.empty:
        raise ValueError(
            "Brak kursów w oknie czasowym na wybranym odcinku — "
            "poszerz zakres lub okno."
        )

    related, station_list = slice_related_to_corridor(
        related_full, corridor_indices, full_station_list, lo, hi
    )
    if related.empty or trip_id not in set(related["trip_id"]):
        raise ValueError(
            "Wybrany kurs nie ma stacji w tym zakresie — poszerz odcinek."
        )

    n_same = int(related.loc[related["role"] == "same", "trip_id"].nunique())
    n_opp = int(related.loc[related["role"] == "opposite", "trip_id"].nunique())
    return MareyResult(
        related=related,
        station_list=station_list,
        selected_trip_id=trip_id,
        selected_line=line_code,
        n_same=n_same,
        n_opp=n_opp,
    )


def marey_explorer(
    data: MareyData,
    *,
    selected_date: date,
    line_code: str,
    trip_id: str,
    local_from: int = 0,
    local_to: int | None = None,
    margin_minutes: int = 60,
    kolory: dict[str, str] | None = None,
) -> Any:
    """Drop-in: dane + wybór parametrów → podsumowanie + zakładki.

    To główna fasada do osadzenia w innej komórce / notebooku, gdy znasz
    już ``line_code`` i ``trip_id`` (z własnych widgetów albo stałych).

    Parameters
    ----------
    data
        Wynik ``load_marey_data()``.
    selected_date, line_code, trip_id
        Wybór dnia, linii KM i kursu GTFS.
    local_from, local_to
        Indeksy wzdłuż korytarza kursu (0 = pierwsza stacja kursu na linii).
        ``local_to=None`` → do końca korytarza.
    margin_minutes
        Okno czasowe ± minut wokół wybranego kursu.
    kolory
        Opcjonalna paleta; domyślnie ``DEFAULT_KOLORY_LINII_KML``.

    Returns
    -------
    marimo Html
        ``mo.vstack([summary, tabs])``. Przy błędzie — markdown z komunikatem.

    Examples
    --------
    ```python
    from marey_viz import load_marey_data, marey_explorer, marey_trip_options

    data = load_marey_data()
    trips = marey_trip_options(data, date(2026, 10, 23), "SKA3P")
    trip_id = next(iter(trips.values()))
    marey_explorer(
        data,
        selected_date=date(2026, 10, 23),
        line_code="SKA3P",
        trip_id=trip_id,
    )
    ```
    """
    try:
        result = marey_compute(
            data,
            selected_date=selected_date,
            line_code=line_code,
            trip_id=trip_id,
            local_from=local_from,
            local_to=local_to,
            margin_minutes=margin_minutes,
        )
    except ValueError as exc:
        return mo.md(f"**Brak widoku:** {exc}")

    sel = result.related[result.related["trip_id"] == trip_id]
    short = sel["trip_short_name"].iloc[0] if len(sel) else trip_id
    panel = marey_panel(
        result.related,
        result.station_list,
        data.line_geoms,
        data.railway,
        selected_line=line_code,
        kolory=kolory,
        chart_title=f"{short} {line_code}",
    )
    return mo.vstack([marey_summary(result), panel], gap=1)


def marey_explorer_form(
    data: MareyData,
    *,
    selected_date: date,
    default_line: str | None = None,
    default_trip_label: str | None = None,
    margin_minutes: int = 60,
) -> Any:
    """Formularz UI (linia, kurs, zakres stacji, okno) dla **ustalonej daty**.

    W marimo kaskada data→kurs wymaga osobnej komórki z ``date_picker``;
    ten formularz przebuduj w komórce zależnej od ``date_picker.value``.

    Po zatwierdzeniu ``form.value`` to dict::

        {
            "line_code": str,
            "trip_label": str,   # klucz z marey_trip_options
            "station_from": int, # lokalny indeks
            "station_to": int,
            "margin_minutes": int,
        }

    Przekaż go do ``marey_from_form``.

    Parameters
    ----------
    data
        Dane z ``load_marey_data``.
    selected_date
        Dzień rozkładu (zwykle ``date_picker.value``).
    default_line, default_trip_label
        Preferowane wartości startowe, jeśli nadal dostępne.
    margin_minutes
        Domyślne okno czasowe na sliderze.
    """
    lines = marey_line_options(data, selected_date)
    if not lines:
        return mo.md("Brak kursów w `calendar_dates` dla wybranej daty.")

    line0 = (
        default_line
        if default_line in lines
        else ("SKA3P" if "SKA3P" in lines else lines[0])
    )
    # Opcje kursów / stacji dla default linii — pełna kaskada linia→kurs
    # w jednym formularzu nie odświeża się live; bierzemy unię kursów dnia
    # z prefiksem linii w etykiecie, żeby jeden dropdown wystarczył.
    day = day_schedule(data.schedule, data.calendar_dates, selected_date)
    trip_groups = {tid: g for tid, g in day.groupby("trip_id", sort=False)}
    trip_ids_sorted = sorted(
        trip_groups,
        key=lambda tid: (
            str(trip_groups[tid]["line_code"].iloc[0]),
            int(trip_groups[tid]["departure_seconds"].min()),
        ),
    )
    trip_opts: dict[str, str] = {}
    for tid in trip_ids_sorted:
        g = trip_groups[tid]
        code = str(g["line_code"].iloc[0])
        trip_opts[f"{code} · {trip_label(g)}"] = tid

    if not trip_opts:
        return mo.md("Brak kursów dla wybranej daty.")

    default_trip = default_trip_label if default_trip_label in trip_opts else next(
        iter(trip_opts)
    )
    # Zakres stacji: po wyborze kursu i tak przycinamy w marey_from_form;
    # tu slider 0–100% korytarza jako przybliżenie (0 = start, 100 = koniec).
    return (
        mo.md(
            """
            **Linia** (używana do osi / mapy; kurs może być innej linii w oknie)

            {line_code}

            **Kurs**

            {trip_label}

            **Zakres korytarza kursu** (0% = początek, 100% = koniec)

            {corridor_pct}

            **Okno czasowe ± min**

            {margin_minutes}
            """
        )
        .batch(
            line_code=mo.ui.dropdown(
                options=lines, value=line0, label="Linia (oś / mapa)"
            ),
            trip_label=mo.ui.dropdown(
                options=list(trip_opts.keys()),
                value=default_trip,
                label="Kurs",
            ),
            corridor_pct=mo.ui.range_slider(
                start=0,
                stop=100,
                step=1,
                value=(0, 100),
                label="Zakres %",
                show_value=True,
            ),
            margin_minutes=mo.ui.slider(
                start=30,
                stop=180,
                step=15,
                value=margin_minutes,
                label="± min",
                show_value=True,
            ),
        )
        .form(
            submit_button_label="Pokaż wykres / mapę",
            bordered=True,
            clear_on_submit=False,
        )
    )


def marey_from_form(
    data: MareyData,
    form_value: dict[str, Any] | None,
    *,
    selected_date: date,
    kolory: dict[str, str] | None = None,
) -> Any:
    """Render drop-in z wartości formularza ``marey_explorer_form``.

    Parameters
    ----------
    data
        Dane wejściowe.
    form_value
        ``form.value`` — ``None`` dopóki użytkownik nie zatwierdzi.
    selected_date
        Ta sama data co przy budowie formularza.
    kolory
        Opcjonalna paleta.
    """
    if form_value is None:
        return mo.md(
            "_Ustaw parametry powyżej i kliknij **Pokaż wykres / mapę**._"
        )

    day = day_schedule(data.schedule, data.calendar_dates, selected_date)
    trip_groups = {tid: g for tid, g in day.groupby("trip_id", sort=False)}
    trip_opts: dict[str, str] = {}
    for tid, g in trip_groups.items():
        code = str(g["line_code"].iloc[0])
        trip_opts[f"{code} · {trip_label(g)}"] = tid

    trip_key = form_value["trip_label"]
    if trip_key not in trip_opts:
        return mo.md("Wybrany kurs nie istnieje w tym dniu — odśwież formularz.")

    trip_id = trip_opts[trip_key]
    trip_line = str(trip_groups[trip_id]["line_code"].iloc[0])
    # Oś mapy/wykresu: preferuj linię z formularza, jeśli kurs da się na nią
    # zmapować; inaczej linię własną kursu.
    line_code = form_value["line_code"]
    if line_code not in data.stations_by_line:
        line_code = trip_line

    corridor = trip_corridor_indices(
        day, trip_id, data.stations_by_line[line_code]
    )
    if len(corridor) < 2:
        # Fallback: oś = linia kursu
        line_code = trip_line
        corridor = trip_corridor_indices(
            day, trip_id, data.stations_by_line[line_code]
        )
    if len(corridor) < 2:
        return mo.md("Ten kurs nie da się zmapować na stacje linii.")

    pct_lo, pct_hi = form_value["corridor_pct"]
    n = len(corridor)
    local_from = int(round((pct_lo / 100) * (n - 1)))
    local_to = int(round((pct_hi / 100) * (n - 1)))

    return marey_explorer(
        data,
        selected_date=selected_date,
        line_code=line_code,
        trip_id=trip_id,
        local_from=local_from,
        local_to=local_to,
        margin_minutes=int(form_value["margin_minutes"]),
        kolory=kolory,
    )
