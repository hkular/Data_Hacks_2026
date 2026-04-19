#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:11:33 2026

@author: hollykular
"""


"""
California Asthma & Air Quality Dashboard
==========================================
Interactive choropleth map with metric switcher, year slider, and animation.

"""

import requests
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, ctx
from pathlib import Path
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    # Get the directory of the current script
    script_dir = Path(os.path.abspath(__file__)).parent
    # Go up one level to the main_repo
    BASE_DIR = script_dir.parent
except NameError:
    # Fallback: IDE working directory
    BASE_DIR = Path(os.getcwd()).parent


df_death      = pd.read_csv(f"{BASE_DIR}/Asthma_Data/cleaned_deaths.csv")
df_er         = pd.read_csv(f"{BASE_DIR}/Asthma_Data/cleaned_er.csv")
df_prevalence = pd.read_csv(f"{BASE_DIR}/Asthma_Data/cleaned_prevalence.csv")
df_aqi        = pd.read_csv(f"{BASE_DIR}/EPA_data/air_aqi_and_particles_annual/aqi_2015-2022.csv")
df_pollutants = pd.read_csv(f"{BASE_DIR}/EPA_data/air_aqi_and_particles_annual/2015-2022.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. NORMALISE → standard shape: {county, year, fips, value}
# ─────────────────────────────────────────────────────────────────────────────

CA_FIPS = {
    "Alameda":"06001","Alpine":"06003","Amador":"06005","Butte":"06007",
    "Calaveras":"06009","Colusa":"06011","Contra Costa":"06013","Del Norte":"06015",
    "El Dorado":"06017","Fresno":"06019","Glenn":"06021","Humboldt":"06023",
    "Imperial":"06025","Inyo":"06027","Kern":"06029","Kings":"06031",
    "Lake":"06033","Lassen":"06035","Los Angeles":"06037","Madera":"06039",
    "Marin":"06041","Mariposa":"06043","Mendocino":"06045","Merced":"06047",
    "Modoc":"06049","Mono":"06051","Monterey":"06053","Napa":"06055",
    "Nevada":"06057","Orange":"06059","Placer":"06061","Plumas":"06063",
    "Riverside":"06065","Sacramento":"06067","San Benito":"06069",
    "San Bernardino":"06071","San Diego":"06073","San Francisco":"06075",
    "San Joaquin":"06077","San Luis Obispo":"06079","San Mateo":"06081",
    "Santa Barbara":"06083","Santa Clara":"06085","Santa Cruz":"06087",
    "Shasta":"06089","Sierra":"06091","Siskiyou":"06093","Solano":"06095",
    "Sonoma":"06097","Stanislaus":"06099","Sutter":"06101","Tehama":"06103",
    "Trinity":"06105","Tulare":"06107","Tuolumne":"06109","Ventura":"06111",
    "Yolo":"06113","Yuba":"06115",
}

def add_fips(df, county_col="county"):
    """Strip ' County' suffix, title-case, map to FIPS."""
    df = df.copy()
    df[county_col] = (
        df[county_col]
        .str.replace(r"\s*county\s*", "", case=False, regex=True)
        .str.strip()
        .str.title()
    )
    df["fips"] = df[county_col].map(CA_FIPS)
    return df


# fix years hyphen
def expand_years(df, col="YEARS"):
    years = df[col].str.split("-", expand=True).astype(int)
    df["year_start"] = years[0]
    df["year_end"] = years[1]

    df = df.loc[df.index.repeat(df["year_end"] - df["year_start"] + 1)].copy()
    df["year"] = df.groupby(level=0).cumcount() + df["year_start"]

    return df.drop(columns=["year_start", "year_end", col, "YEAR"], errors="ignore")

def clean_years(df, col="YEARS"):
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[^\d]", "-", regex=True)   # replace ANY non-digit with "-"
        .str.replace(r"-+", "-", regex=True)      # collapse multiple dashes
        .str.strip("-")                           # remove leading/trailing dashes
    )
    return df

# load prediction data after fips def
try:
    df_predicted = pd.read_csv(f"{BASE_DIR}/Asthma_Data/predicted.csv")
    # Ensure fips exists in predicted.csv
    if "fips" not in df_predicted.columns:
        df_predicted = add_fips(df_predicted)
except:
    # Placeholder for testing
    df_predicted = pd.DataFrame(columns=["county", "fips", "age_group", "risk_score"])


# ── Deaths ──────────────────────────────────────────────────────────────────
# Has AGE column — keep only total/all-ages rows to avoid double-counting.
# Check what values exist with: print(df_death["AGE"].unique())
death = df_death.copy()
death = clean_years(death, 'YEARS')
death = expand_years(death)
death.columns = death.columns.str.strip()       
death = death.rename(columns={"COUNTY": "county", "YEAR": "year",
                               "AGE-ADJUSTED MORTALITY RATE": "value"})

death = add_fips(death)[["county", "year", "fips", "value"]].dropna(subset=["fips", "value"])
death = (death.groupby(["county", "year", "fips"], as_index=False)["value"].mean())
# ── ER visits ────────────────────────────────────────────────────────────────
er = df_er.copy()
er.columns = er.columns.str.strip()

er = er.rename(columns={"COUNTY": "county", "YEAR": "year",
                         "AGE_ADJUSTED_ED_VISIT_RATE": "value"})


er = add_fips(er)[["county", "year", "fips", "value"]].dropna(subset=["fips", "value"])
er = (er.groupby(["county", "year", "fips"], as_index=False)["value"].mean())
# ── Prevalence ───────────────────────────────────────────────────────────────
# Also has AGE column — same filter as deaths.
prev = df_prevalence.copy()
prev = clean_years(prev, 'YEARS')
prev = expand_years(prev)
prev.columns = prev.columns.str.strip()
prev = prev.rename(columns={"COUNTY": "county", "YEAR": "year",
                              "CURRENT PREVALENCE": "value"})

prev = add_fips(prev)[["county", "year", "fips", "value"]].dropna(subset=["fips", "value"])
prev = (prev.groupby(["county", "year", "fips"], as_index=False)["value"].mean())
# ── AQI ─────────────────────────────────────────────────────────────────────
# county column already lowercase, year already lowercase
aqi = df_aqi.copy()
aqi.columns = aqi.columns.str.strip()
aqi = aqi.rename(columns={"aqi": "value"})   # county + year already correct
aqi = add_fips(aqi)[["county", "year", "fips", "value"]].dropna(subset=["fips", "value"])

# ── Pollutants ───────────────────────────────────────────────────────────────
poll = df_pollutants.copy()
poll.columns = poll.columns.str.strip()
poll = add_fips(poll).dropna(subset=["fips", "avg_val", "particle"])
particle_names = ['Ozone', 'Nitric oxide (NO)', 'Nitrogen dioxide (NO2)']# subset sorted(poll["particle"].unique().tolist())
particle_dfs = {
    p: (poll[poll["particle"] == p][["county", "year", "fips", "avg_val"]]
        .rename(columns={"avg_val": "value"})
        .copy())
    for p in particle_names
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. METRIC REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

METRICS = {
    "death_rate": {
        "label": "Age-adjusted asthma mortality rate",
        "df": death,
        "unit": "per 100k",
        "colorscale": [[0,"#fff5f0"],[0.5,"#fc8d59"],[1,"#67000d"]],
        "group": "Asthma outcomes",
    },
    "er_visits": {
        "label": "Age-adjusted asthma ER visit rate",
        "df": er,
        "unit": "per 100k",
        "colorscale": [[0,"#f7fbff"],[0.5,"#6baed6"],[1,"#08306b"]],
        "group": "Asthma outcomes",
    },
    "prevalence": {
        "label": "Current asthma prevalence",
        "df": prev,
        "unit": "%",
        "colorscale": [[0,"#f7fcfd"],[0.5,"#66c2a4"],[1,"#00441b"]],
        "group": "Asthma outcomes",
    },
    "aqi": {
        "label": "AQI (median)",
        "df": aqi,
        "unit": "AQI",
        "colorscale": [
            [0.00,"#00e400"],
            [0.25,"#ffff00"],
            [0.50,"#ff7e00"],
            [0.75,"#ff0000"],
            [1.00,"#8f3f97"],
        ],
        "group": "Air quality",
    },
}

POLLUTANT_COLORS = [
    [[0,"#fff7fb"],[0.5,"#9ebcda"],[1,"#4d004b"]],
    [[0,"#ffffe5"],[0.5,"#fe9929"],[1,"#662506"]],
    [[0,"#f7fcf0"],[0.5,"#74c476"],[1,"#00441b"]],
    [[0,"#fff5eb"],[0.5,"#fd8d3c"],[1,"#7f2704"]],
    [[0,"#f7f4f9"],[0.5,"#9e9ac8"],[1,"#3f007d"]],
]
for i, p in enumerate(particle_names):
    METRICS[f"pollutant_{p}"] = {
        "label": f"Pollutant: {p}",
        "df": particle_dfs[p],
        "unit": "μg/m³",
        "colorscale": POLLUTANT_COLORS[i % len(POLLUTANT_COLORS)],
        "group": "Pollutants",
    }

def build_dropdown_options():
    groups = {}
    for key, m in METRICS.items():
        groups.setdefault(m["group"], []).append({"label": m["label"], "value": key})
    options = []
    for group, opts in groups.items():
        options.append({"label": f"── {group} ──", "value": f"__hdr_{group}", "disabled": True})
        options.extend(opts)
    return options

# ─────────────────────────────────────────────────────────────────────────────
# 4. GEOJSON
# ─────────────────────────────────────────────────────────────────────────────

print("Loading GeoJSON…")
resp    = requests.get(
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    timeout=20,
)
full_geo = resp.json()
ca_fips_set = set(CA_FIPS.values())
ca_geo = {
    "type": "FeatureCollection",
    "features": [f for f in full_geo["features"] if f["id"] in ca_fips_set],
}
print(f"GeoJSON loaded — {len(ca_geo['features'])} CA counties")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

CARD  = {"background":"#fff","borderRadius":"10px","border":"0.5px solid #e0e0e0","padding":"18px 20px"}
LABEL = {"fontSize":"11px","color":"#888","fontFamily":"sans-serif",
         "textTransform":"uppercase","letterSpacing":"0.5px","display":"block","marginBottom":"6px"}

#app = Dash(__name__)
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server 
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)

app.layout = html.Div(
    style={"fontFamily":"Georgia, serif","backgroundColor":"#f9f8f5",
           "minHeight":"100vh","padding":"24px 32px", "overflowY": "auto"},
    children=[
        # ── NEW SECTION: PREDICTION TOP ──────────────────────────────────────
        html.H1("Asthma Risk Predictor", 
                style={"margin":"0 0 4px","fontSize":"26px","fontWeight":"500","color":"#1a1a1a"}),
        html.P("Predictive modeling based on demographic and environmental factors.",
               style={"margin":"0 0 20px","color":"#777","fontSize":"13px","fontFamily":"sans-serif"}),
        
        html.Div(style={**CARD, "marginBottom":"40px", "display":"grid", "gridTemplateColumns":"300px 1fr", "gap":"24px"}, children=[
            # Prediction Controls
            html.Div([
                html.Label("Select Age Group", style=LABEL),
                dcc.Dropdown(id="pred-age-select", 
                             options=[{"label": "Adult", "value": "adult"}, {"label": "Child", "value": "child"}],
                             value="adult", clearable=False, style={"marginBottom":"20px", "fontFamily":"sans-serif"}),
                
                html.Label("Select County", style=LABEL),
                dcc.Dropdown(id="pred-county-select", 
                             options=[{"label": c, "value": c} for c in sorted(CA_FIPS.keys())],
                             value="Alameda", clearable=False, style={"fontFamily":"sans-serif"}),
                
                html.Div(id="risk-score-display", style={"marginTop":"40px"})
            ]),
            
            # Prediction Map
            dcc.Graph(id="pred-map", style={"height":"450px"})
        ]),

        html.Hr(style={"border":"0","borderTop":"1px solid #e0e0e0","margin":"40px 0"}),

        # ── EXISTING DASHBOARD SECTION ────────────────────────────────────────
            # Header
            html.H1("California Asthma & Air Quality Dashboard",
                    style={"margin":"0 0 4px","fontSize":"26px","fontWeight":"500",
                           "color":"#1a1a1a","letterSpacing":"-0.5px"}),
            html.P("County-level data · Select a metric and drag the year slider",
                   style={"margin":"0 0 20px","color":"#777","fontSize":"13px","fontFamily":"sans-serif"}),

            # Controls
            html.Div(style={"display":"flex","gap":"20px","alignItems":"flex-end",
                            "marginBottom":"20px","flexWrap":"wrap"}, children=[
                html.Div(style={"flex":"1","minWidth":"280px"}, children=[
                    html.Label("Metric", style=LABEL),
                    dcc.Dropdown(id="metric-select", options=build_dropdown_options(),
                                 value="death_rate", clearable=False,
                                 style={"fontFamily":"sans-serif","fontSize":"13px"}),
                ]),
                html.Div(style={"flex":"2","minWidth":"300px"}, children=[
                    html.Label("Year", style=LABEL),
                    html.Div(id="slider-wrap"),
                ]),
                html.Div(style={"display":"flex","gap":"8px","paddingBottom":"2px"}, children=[
                    html.Button("▶  Play", id="play-btn",
                        style={"padding":"8px 16px","fontSize":"13px","fontFamily":"sans-serif",
                               "cursor":"pointer","border":"0.5px solid #ccc",
                               "borderRadius":"6px","background":"#fff","whiteSpace":"nowrap"}),
                    dcc.Interval(id="anim-interval", interval=900, disabled=True, n_intervals=0),
                ]),
            ]),
            # Slider

            # Metric cards
            html.Div(id="metric-cards",
                     style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)",
                            "gap":"12px","marginBottom":"20px"}),

            # Map + sidebar
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 300px","gap":"18px"}, children=[
                dcc.Graph(id="choro-map",
                          style={"height":"600px","borderRadius":"10px",
                                 "border":"0.5px solid #e0e0e0","overflow":"hidden"}),
                html.Div([
                    html.Div(id="county-detail", style={**CARD,"marginBottom":"16px"}),
                    html.Div(id="rankings",      style=CARD),
                ]),
            ]),
            
            html.Div(
                dcc.Slider(id="year-slider", min=0, max=1, value=0),
                style={"display": "none"}
            )
                    
    ]
)



# ─────────────────────────────────────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────


@app.callback(
    Output("pred-map", "figure"),
    Output("risk-score-display", "children"),
    Input("pred-age-select", "value"),
    Input("pred-county-select", "value")
)
def update_prediction_map(age_group, selected_county):
    # Filter data for chosen age
    # Assumes your predicted.csv has 'age_group' (adult/child) and 'risk_score'
    filtered_df = df_predicted[df_predicted['age_group'] == age_group].copy()
    
    if filtered_df.empty:
        return px.choropleth(), html.P("No prediction data available")

    # Create a 'selected' column for visual bolding
    # Map uses FIPS, so we find the FIPS of the selected county
    selected_fips = CA_FIPS.get(selected_county)
    
    # Base Map
    fig = px.choropleth(
        filtered_df,
        geojson=ca_geo,
        locations="fips",
        color="risk_score",
        color_continuous_scale="Purples",
        hover_name="county",
        labels={"risk_score": "Risk Level"}
    )
    
    # Highlight logic: Update line width for the selected county
    fig.update_traces(
        marker_line_width=[3 if f == selected_fips else 0.5 for f in filtered_df['fips']],
        marker_line_color=["black" if f == selected_fips else "white" for f in filtered_df['fips']]
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    # Risk Card
    row = filtered_df[filtered_df['county'] == selected_county]
    score = row['risk_score'].values[0] if not row.empty else 0
    
    risk_card = html.Div([
        html.P(f"Predicted Risk for {selected_county}", style=LABEL),
        html.P(f"{score:.1f}", style={"fontSize":"48px", "fontWeight":"bold", "margin":"0", "color":"#4d004b"}),
        html.P("Model-based estimation", style={"fontSize":"12px", "color":"#888"})
    ])

    return fig, risk_card

@app.callback(
    Output("slider-wrap", "children"),
    Input("metric-select", "value"),
)
def build_slider(metric_key):
    years = sorted(
        METRICS[metric_key]["df"]["year"].dropna().unique().astype(int).tolist()
    )
    if not years:
        return html.P("No year data", style={"fontSize":"13px","color":"#888"})
    marks = {y: {"label": str(y), "style": {"fontSize":"11px","fontFamily":"sans-serif"}}
              for y in years}
    return dcc.Slider(id="year-slider", min=years[0], max=years[-1],
                      value=years[-1], marks=marks, step=1,
                      tooltip={"placement":"bottom","always_visible":False})


@app.callback(
    Output("choro-map",    "figure"),
    Output("metric-cards", "children"),
    Output("rankings",     "children"),
    Input("metric-select", "value"),
    Input("year-slider",   "value"),
)
def update_map(metric_key, year):
    if year is None:
        return {}, [], []

    m    = METRICS[metric_key]
    unit = m["unit"]
    lbl  = m["label"]

    year_df = (m["df"][m["df"]["year"] == int(year)]
               .copy()
               .assign(value=lambda d: pd.to_numeric(d["value"], errors="coerce"))
               .dropna(subset=["value","fips"]))

    if year_df.empty:
        fig = px.choropleth()
        fig.update_layout(
            annotations=[{"text":f"No data for {lbl} in {year}","showarrow":False,
                          "font":{"size":14}}],
            paper_bgcolor="#f9f8f5",
        )
        return fig, [], []

    vmin = year_df["value"].quantile(0.02)
    vmax = year_df["value"].quantile(0.98)

    fig = px.choropleth(
        year_df, geojson=ca_geo, locations="fips", color="value",
        color_continuous_scale=m["colorscale"], range_color=[vmin, vmax],
        hover_name="county", hover_data={"fips":False,"value":":.2f"},
        labels={"value": lbl},
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#f9f8f5")
    fig.update_layout(
        margin={"r":0,"t":36,"l":0,"b":0},
        paper_bgcolor="#f9f8f5", plot_bgcolor="#f9f8f5",
        title={"text":f"{lbl}  ·  {year}","font":{"size":14,"family":"Georgia"},
               "x":0.01,"xanchor":"left"},
        coloraxis_colorbar=dict(
            title=dict(text=unit, font=dict(size=11,family="sans-serif")),
            thickness=13, len=0.65,
        ),
    )

    # Cards
    avg   = year_df["value"].mean()
    worst = year_df.loc[year_df["value"].idxmax()]
    best  = year_df.loc[year_df["value"].idxmin()]

    def card(label, val, sub=""):
        return html.Div(style={"background":"#fff","borderRadius":"8px",
                               "border":"0.5px solid #e0e0e0","padding":"14px 16px"}, children=[
            html.P(label, style={"margin":"0 0 5px","fontSize":"11px","color":"#888",
                                  "fontFamily":"sans-serif","textTransform":"uppercase",
                                  "letterSpacing":"0.5px"}),
            html.P(val,   style={"margin":0,"fontSize":"24px","fontWeight":"500","color":"#1a1a1a","lineHeight":"1"}),
            html.P(sub,   style={"margin":"4px 0 0","fontSize":"12px","color":"#1D9E75","fontFamily":"sans-serif"}),
        ])

    cards = [
        card("State average",        f"{avg:.1f}",             unit),
        card("Highest",              f"{worst['value']:.1f}",  worst["county"]),
        card("Lowest",               f"{best['value']:.1f}",   best["county"]),
        card("Counties reporting",   str(len(year_df)),        f"of {len(CA_FIPS)}"),
    ]

    # Rankings
    top5 = year_df.nlargest(5,  "value")[["county","value"]]
    bot5 = year_df.nsmallest(5, "value")[["county","value"]]

    def rank_row(county, val, color):
        return html.Div(style={"display":"flex","justifyContent":"space-between",
                               "padding":"5px 0","borderBottom":"0.5px solid #f0f0f0",
                               "fontFamily":"sans-serif","fontSize":"13px"}, children=[
            html.Span(county, style={"color":"#333"}),
            html.Span(f"{val:.1f}", style={"fontWeight":"500","color":color}),
        ])

    rankings = html.Div([
        html.P("Rankings", style={"margin":"0 0 10px","fontSize":"14px","fontWeight":"500","color":"#1a1a1a"}),
        html.P("Highest", style={"margin":"0 0 5px","fontSize":"11px","color":"#888",
               "fontFamily":"sans-serif","textTransform":"uppercase","letterSpacing":"0.5px"}),
        *[rank_row(r.county, r.value, "#cc3300") for _, r in top5.iterrows()],
        html.Div(style={"height":"12px"}),
        html.P("Lowest", style={"margin":"0 0 5px","fontSize":"11px","color":"#888",
               "fontFamily":"sans-serif","textTransform":"uppercase","letterSpacing":"0.5px"}),
        *[rank_row(r.county, r.value, "#1D9E75") for _, r in bot5.iterrows()],
    ])

    return fig, cards, rankings


@app.callback(
    Output("county-detail", "children"),
    Input("choro-map",      "clickData"),
    Input("metric-select",  "value"),
    Input("year-slider",    "value"),
)
def county_detail(click_data, metric_key, year):
    m    = METRICS[metric_key]
    df   = m["df"]
    unit = m["unit"]

    if not click_data:
        return html.Div([
            html.P("County detail", style={"margin":"0 0 6px","fontSize":"14px","fontWeight":"500"}),
            html.P("Click a county on the map.", style={"fontSize":"13px","color":"#888","fontFamily":"sans-serif"}),
        ])

    county_name = click_data["points"][0]["hovertext"]
    cdf = (df[df["county"] == county_name].copy()
           .assign(value=lambda d: pd.to_numeric(d["value"], errors="coerce"))
           .dropna(subset=["value"])
           .sort_values("year"))

    if cdf.empty:
        return html.P(f"No data for {county_name}.",
                      style={"fontSize":"13px","color":"#888"})

    current_rows = cdf[cdf["year"] == int(year)] if year else pd.DataFrame()
    current_val  = current_rows["value"].values[0] if not current_rows.empty else None
    vals = cdf["value"].tolist()
    trend_dir   = "↓ decreasing" if vals[-1] < vals[0] else "↑ increasing"
    trend_color = "#1D9E75" if "decreasing" in trend_dir else "#cc3300"

    max_v = max(vals) or 1
    spark = [
        html.Div(style={
            "width":"14px",
            "height":f"{max(4, int((row['value']/max_v)*56))}px",
            "background": "#cc3300" if (year and int(row["year"]) == int(year)) else "#e0c4b8",
            "borderRadius":"2px 2px 0 0","alignSelf":"flex-end",
        })
        for _, row in cdf.iterrows()
    ]
    yrs = cdf["year"].tolist()

    return html.Div([
        html.P(county_name, style={"margin":"0 0 2px","fontSize":"17px","fontWeight":"500"}),
        html.P(trend_dir,   style={"margin":"0 0 12px","fontSize":"12px",
               "color":trend_color,"fontFamily":"sans-serif"}),
        *([] if current_val is None else [
            html.Div(style={"background":"#f5f5f0","borderRadius":"6px","padding":"10px 14px",
                            "marginBottom":"12px","display":"inline-block"}, children=[
                html.P(f"{current_val:.2f} {unit}", style={"margin":0,"fontSize":"22px","fontWeight":"500"}),
                html.P(str(year), style={"margin":"2px 0 0","fontSize":"12px","color":"#888","fontFamily":"sans-serif"}),
            ])
        ]),
        html.P("Trend", style={"margin":"0 0 5px","fontSize":"11px","color":"#888",
               "fontFamily":"sans-serif","textTransform":"uppercase","letterSpacing":"0.5px"}),
        html.Div(style={"display":"flex","gap":"2px","alignItems":"flex-end","height":"60px"},
                 children=spark),
        html.Div(style={"display":"flex","justifyContent":"space-between",
                        "fontFamily":"sans-serif","fontSize":"11px","color":"#aaa","marginTop":"3px"},
                 children=[html.Span(str(yrs[0])), html.Span(str(yrs[-1]))]),
    ])


@app.callback(
    Output("year-slider",   "value"),
    Output("anim-interval", "disabled"),
    Output("play-btn",      "children"),
    Input("play-btn",       "n_clicks"),
    Input("anim-interval",  "n_intervals"),
    Input("metric-select",  "value"),
    prevent_initial_call=True,
)
def animate(n_clicks, n_intervals, metric_key):
    triggered = ctx.triggered_id
    years = sorted(
        METRICS[metric_key]["df"]["year"].dropna().unique().astype(int).tolist()
    )
    if not years:
        return 2022, True, "▶  Play"

    if triggered == "play-btn":
        playing = (n_clicks or 0) % 2 == 1
        return years[0], not playing, ("⏸  Pause" if playing else "▶  Play")

    if triggered == "anim-interval":
        idx  = n_intervals % len(years)
        done = idx == len(years) - 1
        return years[idx], done, ("▶  Play" if done else "⏸  Pause")

    return years[-1], True, "▶  Play"


if __name__ == "__main__":
    app.run(debug=True)
