# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pyarrow",
#     "altair",
#     "folium",
#     "geopandas",
# ]
# ///

"""Interaktywny notebook Mareya — wiring UI na bazie ``marey_viz``.

Drop-in do innych notebooków::

    from marey_viz import load_marey_data, marey_explorer, marey_explorer_form

Zobacz docstring modułu ``marey_viz`` (quick start + kontrakt danych).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="large")

with app.setup:
    from datetime import date
    from pathlib import Path
    import sys

    import marimo as mo

    # 1. Ścieżki lokalne (gdy uruchamiasz u siebie z folderu notebooks/)
    _here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for _p in (_here, _here.parent):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    # 2. Fallback chmurowy (molab): jeśli nie ma marey_viz.py ani data/processed, pobierz z GitHuba
    GITHUB_RAW = "https://raw.githubusercontent.com/BartoszKog/km-projekt/main"

    if not (_here / "marey_viz.py").exists() and not (_here.parent / "marey_viz.py").exists():
        urllib.request.urlretrieve(f"{GITHUB_RAW}/marey_viz.py", "marey_viz.py")

    if not Path("data/processed").exists() and not (_here.parent / "data/processed").exists():
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        for _fname in [
            "schedule.parquet",
            "calendar_dates.parquet",
            "stations_lists.parquet",
            "railway_data_clean.parquet",
            "km_line_geometries.parquet",
        ]:
            urllib.request.urlretrieve(
                f"{GITHUB_RAW}/data/processed/{_fname}",
                f"data/processed/{_fname}",
            )

    from marey_viz import (
        DEFAULT_KOLORY_LINII_KML,
        corridor_dropdown_options,
        day_schedule,
        load_marey_data,
        marey_explorer,
        marey_trip_options,
        trip_corridor_indices,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Wykres ruchu kolejowego — Koleje Małopolskie

    Wybierz **datę**, **linię** i **kurs**, potem **zakres stacji wzdłuż tego kursu**
    (kolejność = kierunek jazdy).
    """)
    return


@app.cell
def load_data():
    data = load_marey_data()
    return (data,)


@app.cell
def controls(data):
    available_dates = data.available_dates
    preferred = date(2026, 10, 23)
    default_date = (
        preferred
        if preferred in available_dates
        else available_dates[len(available_dates) // 2]
    )
    date_picker = mo.ui.date(
        value=default_date,
        label="Data",
        start=available_dates[0],
        stop=available_dates[-1],
    )
    margin_slider = mo.ui.slider(
        start=30,
        stop=180,
        step=15,
        value=60,
        label="Okno czasowe ± min",
        show_value=True,
    )
    mo.hstack([date_picker, margin_slider], justify="start", gap=2)
    return date_picker, margin_slider


@app.cell
def _(data, date_picker):
    day = day_schedule(data.schedule, data.calendar_dates, date_picker.value)
    line_options = sorted(day["line_code"].dropna().unique().tolist())
    mo.stop(
        not line_options,
        mo.md("Brak kursów w `calendar_dates` dla wybranej daty."),
    )
    return day, line_options


@app.cell
def _(line_options):
    default_line = "SKA3P" if "SKA3P" in line_options else line_options[0]
    line_dropdown = mo.ui.dropdown(
        options=line_options,
        value=default_line,
        label="Linia",
    )
    line_dropdown
    return (line_dropdown,)


@app.cell
def _(data, line_dropdown):
    full_station_list = data.stations_by_line[line_dropdown.value]
    return (full_station_list,)


@app.cell
def _(data, date_picker, line_dropdown):
    trip_options = marey_trip_options(
        data, date_picker.value, line_dropdown.value
    )
    mo.stop(
        not trip_options,
        mo.md(f"Brak kursów linii **{line_dropdown.value}** w tym dniu."),
    )
    return (trip_options,)


@app.cell
def _(trip_options):
    trip_labels = list(trip_options.keys())
    trip_dropdown = mo.ui.dropdown(
        options=trip_labels,
        value=trip_labels[0],
        label="Kurs (godzina odjazdu)",
    )
    trip_dropdown
    return (trip_dropdown,)


@app.cell
def _(day, full_station_list, trip_dropdown, trip_options):
    mo.stop(
        trip_dropdown.value not in trip_options,
        mo.md("Wybierz kurs z listy."),
    )
    selected_trip_id = trip_options[trip_dropdown.value]
    corridor_indices = trip_corridor_indices(
        day, selected_trip_id, full_station_list
    )
    mo.stop(
        len(corridor_indices) < 2,
        mo.md("Ten kurs nie da się zmapować na stacje linii."),
    )
    from_options = corridor_dropdown_options(corridor_indices, full_station_list)
    station_from = mo.ui.dropdown(
        options=from_options,
        value=next(iter(from_options)),
        label="Od stacji (wzdłuż kursu)",
    )
    return corridor_indices, selected_trip_id, station_from


@app.cell
def _(corridor_indices, full_station_list, station_from):
    local_from = int(station_from.value)
    to_options = {
        f"{local:02d} · {full_station_list[corridor_indices[local]]}": local
        for local in range(local_from, len(corridor_indices))
    }
    station_to = mo.ui.dropdown(
        options=to_options,
        value=next(reversed(to_options)),
        label="Do stacji (wzdłuż kursu)",
    )
    mo.hstack([station_from, station_to], justify="start", gap=2)
    return (station_to,)


@app.cell
def _(
    data,
    date_picker,
    line_dropdown,
    margin_slider,
    selected_trip_id,
    station_from,
    station_to,
):
    view = marey_explorer(
        data,
        selected_date=date_picker.value,
        line_code=line_dropdown.value,
        trip_id=selected_trip_id,
        local_from=int(station_from.value),
        local_to=int(station_to.value),
        margin_minutes=int(margin_slider.value),
        kolory=DEFAULT_KOLORY_LINII_KML,
    )
    view
    return


if __name__ == "__main__":
    app.run()
