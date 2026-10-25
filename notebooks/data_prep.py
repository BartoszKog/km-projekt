import marimo

__generated_with = "0.25.2"
app = marimo.App()

with app.setup:
    import marimo as mo

    from pathlib import Path
    from typing import Any, cast
    from zipfile import ZipFile

    import pandas as pd
    import geopandas as gpd

    from datetime import date


@app.cell(hide_code=True)
def gtfs_intro():
    mo.md(f"""
    # Koleje Małopolskie GTFS - obróbka danych
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Koleje

    Wczytano feed z `{GTFS_ZIP.as_posix()}`
    """)
    return


@app.cell
def load_gtfs():
    GTFS_ZIP = Path("data/raw/kml-ska-gtfs.zip")


    def read_gtfs(name: str) -> pd.DataFrame:
        with ZipFile(GTFS_ZIP) as zf:
            with zf.open(name) as f:
                return pd.read_csv(f)


    agency = read_gtfs("agency.txt")
    routes = read_gtfs("routes.txt")
    stops = read_gtfs("stops.txt")
    trips = read_gtfs("trips.txt")
    stop_times = read_gtfs("stop_times.txt")

    calendar_dates = read_gtfs("calendar_dates.txt")
    calendar_dates['date'] = calendar_dates['date'].apply(
        lambda x: date(
            int(str(x)[:4]), 
            int(str(x)[4:6]), 
            int(str(x)[6:])
            )
        )
    return agency, calendar_dates, routes, stop_times, stops, trips


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wczytane ramki danych
    """)
    return


@app.cell
def _(agency):
    agency
    return


@app.cell
def _(routes):
    routes
    return


@app.cell
def _(stops):
    stops
    return


@app.cell
def _(trips):
    trips
    return


@app.cell
def _(stop_times):
    stop_times
    return


@app.cell
def _(calendar_dates):
    calendar_dates
    return


@app.cell
def _(calendar_dates):
    # zakres dat
    min_date = calendar_dates['date'].min()
    max_date = calendar_dates['date'].max()
    mo.md(text=f"Zakres dat: {min_date} - {max_date}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Kody linii (SKA / K / M)

    Numeracja wg systemu woj. małopolskiego (od 15.06.2025):
    **SKA** aglomeracja, **K** regionalne, **M** transgraniczne; sufiks **P** = przyspieszony.
    """)
    return


@app.function
def assign_line_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaj kolumnę `line_code` do ramki rozkładu.

    Wymaga kolumn: trip_id, stop_sequence, stop_name, trip_short_name.
    Kod jest liczony per trip i powielany na wszystkie wiersze kursu.
    """
    # Relacje: https://kolejemalopolskie.com.pl/pl/rozklad-jazdy/rozklady-kolejowe
    # Schemat: .../2025/06/05/aeitw5/nowy-schemat-linie-kolejowe.pdf
    NAMED_LINE: dict[str, str] = {
        "LUXTORPEDA": "K5P",
        "HUBAL": "K71P",
        "JASKÓŁKA": "SKA3P",
        "MOŚCICKI": "SKA3P",
    }

    KRK = "KRAKÓW GŁÓWNY"
    LOTNISKO = "KRAKÓW LOTNISKO"
    WIELICZKA = "WIELICZKA RYNEK-KOPALNIA"
    NOWA_HUTA = "KRAKÓW NOWA HUTA"
    NOWY_SACZ = "NOWY SĄCZ"
    KRYNICA = "KRYNICA-ZDRÓJ"

    SKA1_ENDS = {LOTNISKO, WIELICZKA}
    SKA2_NORTH = {"SĘDZISZÓW", "MIECHÓW", "KOZŁÓW", "SŁOMNIKI"}
    K71_MARKERS = {"JASŁO", "GORLICE ZAGÓRZANY", "BIECZ"}
    K7_SOUTH = {NOWY_SACZ, KRYNICA, "STRÓŻE", "GRYBÓW"}
    K63_ENDS = {KRK, NOWA_HUTA, "PODŁĘŻE"}

    def _paced(n_stops: int, limit: int, code: str) -> str:
        return f"{code}P" if n_stops <= limit else code

    def _paced_by_max_sequence(max_sequence: int, limit: int, code: str) -> str:
        return f"{code}P" if max_sequence <= limit else code

    def code_from_path(
        start: str,
        end: str,
        n_stops: int,
        max_stop_sequence: int,
        stop_set: set[str],
        train_name: str,
    ) -> str:
        ends = {start, end}

        named = NAMED_LINE.get(train_name)
        if named:
            return named

        # SKA1: Lotnisko <-> Wieliczka (+ skroty przez Krakow Glowny)
        if ends & SKA1_ENDS:
            return "SKA1"
        if LOTNISKO in stop_set and WIELICZKA in stop_set:
            return "SKA1"

        # K5: Zakopane
        if "ZAKOPANE" in ends or "ZAKOPANE" in stop_set:
            return _paced(n_stops, 16, "K5")

        # K71: Jaslo / Gorlice (przed dalszym ciagiem na K7)
        on_k71 = bool(ends & K71_MARKERS) or "JASŁO" in stop_set
        if on_k71:
            if KRK in ends or "TARNÓW" in stop_set:
                return _paced(n_stops, 34, "K71")
            return "K71"

        # K7: Nowy Sacz / Krynica (ew. przez Tarnow)
        on_k7 = bool(ends & K7_SOUTH) or bool(
            stop_set & {NOWY_SACZ, KRYNICA}
        )
        if on_k7:
            # K7P: ostatni stop_sequence 34; K7 zwykly: 52
            return _paced_by_max_sequence(max_stop_sequence, 34, "K7")

        # K63: krotki odcinek Krakow Glowny - Nowa Huta / Podleze
        if ends <= K63_ENDS and n_stops <= 8:
            return "K63"

        via_skawina = "SKAWINA" in stop_set
        via_trzebinia = "TRZEBINIA" in stop_set
        via_miechow = "MIECHÓW" in stop_set or bool(ends & SKA2_NORTH)

        # SKA2 zachod: przez Skawine do Oswiecimia / Zatora
        ska2_west = via_skawina and (
            "OŚWIĘCIM" in ends
            or "ZATOR" in ends
            or "OŚWIĘCIM" in stop_set
            or ends == {KRK, "SKAWINA"}
        )
        if ska2_west:
            return "SKA2"

        # SKA2 polnoc: Miechow / Sedziszow / Kozlow
        if via_miechow or bool(ends & SKA2_NORTH):
            return "SKA2"

        # SKA3 zachod: przez Trzebinie do Oswiecimia
        if via_trzebinia and (
            "OŚWIĘCIM" in ends or "OŚWIĘCIM" in stop_set
        ):
            return "SKA3"

        # SKA3 wschod: Krakow - Tarnow (bez dalszego ciagu na K7)
        # przyspieszone ~13 postojow vs zwykle 22-24
        if "TARNÓW" in ends and not (stop_set & K7_SOUTH):
            return _paced(n_stops, 16, "SKA3")
        if "TARNÓW" in stop_set and KRK in ends:
            return _paced(n_stops, 16, "SKA3")

        return "UNKNOWN"

    required = {
        "trip_id",
        "stop_sequence",
        "stop_name",
        "trip_short_name",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak kolumn: {sorted(missing)}")

    st = df.sort_values(["trip_id", "stop_sequence"])
    trip_meta = (
        st.groupby("trip_id", sort=False)
        .agg(
            start_stop=("stop_name", "first"),
            end_stop=("stop_name", "last"),
            n_stops=("stop_name", "size"),
            max_stop_sequence=("stop_sequence", "max"),
            stop_set=("stop_name", lambda s: set(s.dropna())),
            trip_short_name=("trip_short_name", "first"),
        )
        .reset_index()
    )
    trip_meta["train_name"] = (
        trip_meta["trip_short_name"]
        .fillna("")
        .astype(str)
        .str.replace(r"^[\d/]+\s*", "", regex=True)
        .str.strip()
    )
    trip_meta["line_code"] = [
        code_from_path(
            row.start_stop,
            row.end_stop,
            int(row.n_stops),
            int(row.max_stop_sequence),
            row.stop_set,
            row.train_name,
        )
        for row in cast(Any, trip_meta.itertuples(index=False))
    ]
    codes = trip_meta.set_index("trip_id")["line_code"]
    return df.assign(line_code=df["trip_id"].map(codes))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Rozklad (stop_times + stops + trips)
    """)
    return


@app.function
def gtfs_time_to_seconds(series: pd.Series) -> pd.Series:
    parts = series.str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


@app.cell
def schedule(stop_times, stops, trips):
    schedule = (
        stop_times.drop(columns=["pickup_type", "drop_off_type"])
        .merge(
            stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
            on="stop_id",
            how="left",
        )
        .merge(
            trips[["trip_id", "route_id", "service_id", "trip_short_name"]],
            on="trip_id",
            how="left",
        )
        .assign(
            arrival_seconds=lambda df: gtfs_time_to_seconds(df["arrival_time"]),
            departure_seconds=lambda df: gtfs_time_to_seconds(df["departure_time"]),
        )
    )
    schedule = assign_line_codes(schedule)
    schedule
    return (schedule,)


@app.cell
def _(schedule):
    mo.ui.dataframe(schedule)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Kolejność stacji
    """)
    return


@app.cell
def _(schedule):
    # Utwórz ramkę danych z line_code oraz listą przystanków w kolejności (dla tras, gdzie stop_sequence kończy się na największej liczbie)
    # Załóżmy, że 'schedule' jest dostępny.

    # Najwyższy stop_sequence dla każdej linii (line_code) oraz trip_id
    last_stops = (
        schedule.groupby(["line_code", "trip_id"])["stop_sequence"].max().reset_index()
    )

    # Wybierz dla każdej linii (line_code) taki trip_id, który ma największy stop_sequence
    idx = last_stops.groupby("line_code")["stop_sequence"].idxmax()
    selected_trips = last_stops.loc[idx][["line_code", "trip_id"]]

    # Dołącz trip_id do schedule i posortuj według stop_sequence
    merged = schedule.merge(selected_trips, on=["line_code", "trip_id"], how="inner")
    merged = merged.sort_values(["line_code", "trip_id", "stop_sequence"])

    # Utwórz listę przystanków (stop_name) dla każdej linii w odpowiedniej kolejności
    stations_lists = (
        merged.groupby("line_code")["stop_name"]
        .apply(list)
        .reset_index()
        .rename(columns={"stop_name": "station_sequence"})
    )

    stations_lists
    return last_stops, stations_lists


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Autobusy
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Wczytano feed z `{BUS_GTFS_ZIP.as_posix()}`
    """)
    return


@app.cell
def load_bus_gtfs():
    BUS_GTFS_ZIP = Path("data/raw/ald-gtfs.zip")

    def read_bus_gtfs(name: str) -> pd.DataFrame:
        with ZipFile(BUS_GTFS_ZIP) as zf:
            with zf.open(name) as f:
                return pd.read_csv(f)

    bus_agency = read_bus_gtfs("agency.txt")
    bus_routes = read_bus_gtfs("routes.txt")
    bus_stops = read_bus_gtfs("stops.txt")
    bus_trips = read_bus_gtfs("trips.txt")
    bus_stop_times = read_bus_gtfs("stop_times.txt")

    bus_calendar = read_bus_gtfs("calendar.txt")
    for date_col in ("start_date", "end_date"):
        bus_calendar[date_col] = bus_calendar[date_col].apply(
            lambda x: date(
                int(str(x)[:4]),
                int(str(x)[4:6]),
                int(str(x)[6:]),
            )
        )

    bus_calendar_dates = read_bus_gtfs("calendar_dates.txt")
    bus_calendar_dates["date"] = bus_calendar_dates["date"].apply(
        lambda x: date(
            int(str(x)[:4]),
            int(str(x)[4:6]),
            int(str(x)[6:]),
        )
    )
    return (
        bus_agency,
        bus_calendar,
        bus_calendar_dates,
        bus_routes,
        bus_stop_times,
        bus_stops,
        bus_trips,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wczytane ramki danych (autobusy)
    """)
    return


@app.cell
def _(bus_agency):
    bus_agency
    return


@app.cell
def _(bus_routes):
    bus_routes
    return


@app.cell
def _(bus_stops):
    bus_stops
    return


@app.cell
def _(bus_trips):
    bus_trips
    return


@app.cell
def _(bus_stop_times):
    bus_stop_times
    return


@app.cell
def _(bus_calendar):
    bus_calendar
    return


@app.cell
def _(bus_calendar_dates):
    bus_calendar_dates
    return


@app.cell
def _(bus_calendar_dates):
    bus_min_date = bus_calendar_dates["date"].min()
    bus_max_date = bus_calendar_dates["date"].max()
    mo.md(text=f"Zakres dat (wyjątki): {bus_min_date} - {bus_max_date}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Rozklad autobusow (stop_times + stops + trips + routes)
    """)
    return


@app.cell
def bus_schedule(bus_routes, bus_stop_times, bus_stops, bus_trips):
    bus_schedule = (
        bus_stop_times.drop(columns=["stop_headsign"])
        .merge(
            bus_stops.drop(
                columns=["stop_url", "location_type", "parent_station"]
            ),
            on="stop_id",
            how="left",
        )
        .merge(
            bus_trips.drop(columns=["wheelchair_accessible"]),
            on="trip_id",
            how="left",
        )
        .merge(
            bus_routes[["route_id", "route_short_name"]],
            on="route_id",
            how="left",
        )
        .assign(
            arrival_seconds=lambda df: gtfs_time_to_seconds(df["arrival_time"]),
            departure_seconds=lambda df: gtfs_time_to_seconds(
                df["departure_time"]
            ),
        )
    )
    bus_schedule
    return (bus_schedule,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dane torów kolejowych
    """)
    return


@app.cell
def load_railway_data():
    railway_data = gpd.read_parquet("data/geospatial/railway.parquet")
    return (railway_data,)


@app.cell
def _(railway_data):
    railway_data
    return


@app.cell
def _(railway_data):
    # procent braków danych w kolumnach
    missing_data = railway_data.isnull().mean()
    missing_data
    return


@app.cell
def _(railway_data):
    railway_data.columns
    return


@app.cell
def _(railway_data):
    mo.ui.dataframe(railway_data)
    return


@app.cell
def _(railway_data):
    railway_data_clean = railway_data
    # Select columns first
    railway_data_clean = railway_data_clean[
        [
            "geometry",
            "ref",         # int
            "name",
            "usage",
            "service",
            "maxspeed",    # int
            "passenger_lines", # int
            "electrified",
            "gauge",       # int
        ]
    ]

    # Ustaw odpowiednie typy dla kolumn z komentarzem "int"
    # (gdyby były już int, nic się nie stanie; jeśli nie, zostaną skonwertowane)
    railway_data_clean["ref"] = pd.to_numeric(railway_data_clean["ref"], errors="coerce").astype("Int64")
    railway_data_clean["maxspeed"] = pd.to_numeric(railway_data_clean["maxspeed"], errors="coerce").astype("Int64")
    railway_data_clean["passenger_lines"] = pd.to_numeric(railway_data_clean["passenger_lines"], errors="coerce").astype("Int64")
    railway_data_clean["gauge"] = pd.to_numeric(railway_data_clean["gauge"], errors="coerce").astype("Int64")

    railway_data_clean = railway_data_clean[
        (railway_data_clean["gauge"] == 1435)
        & (railway_data_clean["service"].isna())
    ]
    railway_data_clean = railway_data_clean[railway_data_clean["ref"].notna()]
    railway_data_clean
    return (railway_data_clean,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Mapa torów i postojów (KML / SKA)
    """)
    return


@app.function
def distinct_line_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    return [f"hsl({int(360 * i / n)}, 70%, 42%)" for i in range(n)]


@app.cell
def railway_stops_map(railway_data_clean, stops):
    import json

    import folium

    rail_wgs84 = railway_data_clean.to_crs(epsg=4326).copy()
    rail_wgs84["_stroke"] = distinct_line_colors(len(rail_wgs84))

    stops_map = stops.drop_duplicates(subset="stop_id")
    center_lat = stops_map["stop_lat"].mean()
    center_lon = stops_map["stop_lon"].mean()

    rail_geojson = json.loads(
        rail_wgs84[["geometry", "_stroke", "name", "ref"]].to_json()
    )

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
    )

    folium.GeoJson(
        rail_geojson,
        name="Tory (1435 mm)",
        style_function=lambda feature: {
            "color": feature["properties"]["_stroke"],
            "weight": 2,
            "opacity": 0.85,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "ref"],
            aliases=["Nazwa:", "Ref:"],
            localize=True,
        ),
    ).add_to(m)

    stops_layer = folium.FeatureGroup(name="Postoje")
    for stop in stops_map.itertuples(index=False):
        folium.CircleMarker(
            location=[stop.stop_lat, stop.stop_lon],
            radius=2,
            color="#dc2626",
            fill=True,
            fill_color="#dc2626",
            fill_opacity=0.9,
            tooltip=stop.stop_name,
        ).add_to(stops_layer)
    stops_layer.add_to(m)

    folium.LayerControl().add_to(m)

    mo.Html(m._repr_html_())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Eksploracja danych
    """)
    return


@app.cell
def _(schedule):
    # unikalne wartości line_code z schedule oraz liczba obiektów dla każdego kodu
    line_code_counts = schedule["line_code"].value_counts().reset_index()
    line_code_counts.columns = ["line_code", "count"]
    line_code_counts
    return


@app.cell
def _(last_stops):
    # Tabela pokazująca dla każdej linii (line_code) unikalne wartości końcowego stop_sequence (listy)

    stop_seq_lists = (
        last_stops.groupby("line_code")["stop_sequence"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
        .rename(columns={"stop_sequence": "last_stop_sequences"})
    )

    stop_seq_lists
    return


@app.cell
def _(schedule):
    mo.ui.dataframe(schedule)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Eksport danych
    """)
    return


@app.cell
def _(
    bus_calendar,
    bus_calendar_dates,
    bus_schedule,
    calendar_dates,
    railway_data_clean,
    schedule,
    stations_lists,
):
    # Eksport ramek do parquet
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    calendar_dates.to_parquet(
        output_dir / "calendar_dates.parquet", index=False
    )

    schedule.to_parquet(output_dir / "schedule.parquet", index=False)

    bus_calendar.to_parquet(output_dir / "bus_calendar.parquet", index=False)

    bus_calendar_dates.to_parquet(
        output_dir / "bus_calendar_dates.parquet", index=False
    )

    bus_schedule.to_parquet(output_dir / "bus_schedule.parquet", index=False)

    railway_data_clean.to_parquet(
        output_dir / "railway_data_clean.parquet", index=False
    )

    stations_lists.to_parquet(output_dir / "stations_lists.parquet", index=False)
    return


if __name__ == "__main__":
    app.run()
