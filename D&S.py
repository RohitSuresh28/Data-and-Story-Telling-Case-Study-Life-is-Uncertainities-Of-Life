# app.py — Streamlit UI wrapper around YOUR EXACT analysis code
# Run: streamlit run app.py
# NOTE: No on-screen CSV upload. The code reads from a path inside the script.
# Put your CSV next to this file as "broad_sector.csv" or change DATA_CSV_PATH below.

import os
import io
import re
import builtins
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# =========================
# Config
# =========================
DATA_CSV_PATH = "broad_sector.csv"  # <-- change if needed

st.set_page_config(page_title="Sector & Tech Entry Analyzer", layout="wide")
st.title("📊 Sector Transitions & Tech Entry Paths")
st.caption("Reads data from code (no on-screen upload). Visualizations are kept exactly as in your code.")

# =========================
# Utilities: direct the original code's print/show to Streamlit
# =========================
def _st_print(*args, **kwargs):
    st.text(" ".join(map(str, args)))

builtins.print = _st_print  # Redirect print() to Streamlit

# Capture matplotlib show() into Streamlit
def _st_plt_show(*args, **kwargs):
    st.pyplot(plt.gcf(), clear_figure=False)

plt.show = _st_plt_show  # Redirect plt.show()

# Capture plotly figure show() into Streamlit
def _plotly_show(self, *args, **kwargs):
    st.plotly_chart(self, use_container_width=True)

go.Figure.show = _plotly_show  # Redirect fig.show()

# =========================
# Data loading from code (no uploader)
# =========================
if os.path.exists(DATA_CSV_PATH):
    df = pd.read_csv(DATA_CSV_PATH, on_bad_lines="skip")
    st.success(f"Loaded data from: {DATA_CSV_PATH}")
else:
    st.error(f"Data file not found: {DATA_CSV_PATH}")
    st.stop()

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🔁 First→Final Flow & Retention (exact code)", "🧭 Tech Entry Paths (exact code)"])

with tab1:
    # --- BEGIN: your code block (kept intact) --------------------------------
    # (We define helper(s) and pre-computed vars AROUND it so your code runs unchanged.)

    # Your code expects: to_period_str_quarter()
    def to_period_str_quarter(s):
        if not s or pd.isna(s):
            return None
        s = str(s).strip()
        m = re.match(r'^[Qq]([1-4])\s+(\d{4})$', s)  # e.g., "Q1 2017"
        if m:
            q, y = m.groups()
            return f"{y}Q{q}"
        return None

    # ----- Your lines start (unchanged) -----
    df['start_quarter_str'] = df['start_date'].apply(to_period_str_quarter)
    df['end_quarter_str'] = df['end_date'].apply(to_period_str_quarter)

    df['start_quarter'] = pd.PeriodIndex([p if p is not None else pd.NaT for p in df['start_quarter_str']], freq='Q')
    df['end_quarter'] = pd.PeriodIndex([p if p is not None else pd.NaT for p in df['end_quarter_str']], freq='Q')

    df['cohort'] = df.groupby('person_id')['start_quarter'].transform('min')
    df['cohort_str'] = df['cohort'].astype(str)

    first = df.sort_values('start_quarter').groupby('person_id').first().reset_index()
    last = df.sort_values('start_quarter').groupby('person_id').last().reset_index()
    sector_flow = pd.merge(
        first[['person_id', 'broad_sector', 'cohort']],
        last[['person_id', 'broad_sector']],
        on='person_id',
        suffixes=('_first', '_last')
    )
    # ----- Your lines end (unchanged) -----

    # Pre-compute variables your Sankey snippet uses later (without altering your code)
    transition_counts = (
        sector_flow.groupby(['broad_sector_first', 'broad_sector_last'])
        .size()
        .reset_index(name='count')
    )
    all_labels = sorted(set(transition_counts['broad_sector_first']).union(set(transition_counts['broad_sector_last'])))
    label_to_index = {lab: i for i, lab in enumerate(all_labels)}
    source = transition_counts['broad_sector_first'].map(label_to_index).tolist()
    target = transition_counts['broad_sector_last'].map(label_to_index).tolist()
    value = transition_counts['count'].tolist()

    # ----- Your lines start (unchanged) -----
    palette = [
        "#6BAED6", "#FD8D3C", "#74C476", "#A1D99B",
        "#9E9AC8", "#FDD0A2", "#E377C2", "#B3DE69",
        "#D62728", "#17BECF"
    ]
    node_colors = [palette[i % len(palette)] for i, _ in enumerate(all_labels)]
    link_colors = [node_colors[label_to_index[s]] for s in transition_counts['broad_sector_first']]

    def hex_to_rgba(hex_color, alpha=0.4):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [hex_to_rgba(c, alpha=0.7) for c in link_colors]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_labels,
            line=dict(color="black", width=0.5),
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])
    fig.update_layout(title_text="Career Transition Flow: First → Final Broad Sector", font_size=12)
    fig.show()

    sector_flow['same_sector'] = sector_flow['broad_sector_first'] == sector_flow['broad_sector_last']
    retention = sector_flow.groupby('cohort')['same_sector'].mean().reset_index()
    retention['cohort'] = retention['cohort'].astype(str)

    plt.figure(figsize=(10, 5))
    plt.plot(retention['cohort'], retention['same_sector'] * 100, marker='o')
    plt.xticks(rotation=45)
    plt.title('% Staying in Same Sector (First → Last Job) by Cohort')
    plt.xlabel('Cohort Start Quarter')
    plt.ylabel('Retention Rate (%)')
    plt.grid(True)

    plt.xticks(retention['cohort'][::8], rotation=60)
    plt.tight_layout()
    plt.show()

    transition_matrix = pd.crosstab(first['broad_sector'], last['broad_sector'])
    transition_percentage = transition_matrix.div(transition_matrix.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_percentage, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': '% of People'})
    plt.title('Sector Switch Flow Heatmap (First → Last Broad Sector)')
    plt.xlabel('Final Sector')
    plt.ylabel('Starting Sector')
    plt.tight_layout()
    plt.show()

    # Retention rate per sector
    retention = pd.Series(dtype=float)
    for sector in transition_matrix.index:
        same = transition_matrix.loc[sector, sector] if sector in transition_matrix.columns else 0
        total = transition_matrix.loc[sector].sum()
        retention[sector] = (same / total) * 100 if total > 0 else 0

    retention.sort_values(ascending=False).plot(kind='bar', figsize=(10,5), color='skyblue')
    plt.title('Sector Retention Rate (% of People Who Stayed in the Same Broad Sector)')
    plt.ylabel('Retention %')
    plt.xlabel('Broad Sector')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    print("Top 5 Most Retentive Sectors:")
    print(retention.sort_values(ascending=False).head(5).round(2))

    print("\nTop 5 Most Volatile Sectors (lowest retention):")
    print(retention.sort_values().head(5).round(2))

    # Sort the data for each person chronologically
    df_sorted = df.sort_values(['person_id', 'start_quarter'])

    # Detect loopers
    def went_back(sector_series):
        return (
            len(sector_series) >= 3 and
            sector_series.iloc[0] != sector_series.iloc[-1] and
            sector_series.iloc[0] == sector_series.iloc[-2]
        )

    loop_flags = df_sorted.groupby('person_id')['broad_sector'].apply(went_back)

    loop_counts = loop_flags.value_counts()
    print("🔁 Looping Pattern Detected:\n", loop_counts)

    loop_df = df_sorted.copy()
    loop_df['looped_back'] = loop_df['person_id'].map(loop_flags)

    first_sector = df_sorted.groupby('person_id').first().reset_index()
    first_sector['looped_back'] = first_sector['person_id'].map(loop_flags)

    loop_sector_counts = first_sector[first_sector['looped_back'] == True]['broad_sector'].value_counts()

    plt.figure(figsize=(10,5))
    loop_sector_counts.sort_values(ascending=False).plot(kind='bar', color='orchid')
    plt.title('Top Sectors People Return To Mid-Career')
    plt.ylabel('Number of People Who Looped Back')
    plt.xlabel('Starting Sector')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    df_sorted = df.sort_values(['person_id', 'start_quarter'])
    df_sorted['job_number'] = df_sorted.groupby('person_id').cumcount() + 1
    df_sorted['total_jobs'] = df_sorted.groupby('person_id')['broad_sector'].transform('count')

    final_jobs = df_sorted.groupby('person_id').last().reset_index()
    late_sector_counts = final_jobs['broad_sector'].value_counts()

    print("📊 Late-career (final job) sector distribution:")
    print(late_sector_counts)
    # --- END: your code block ---------------------------------------------------


with tab2:
    # This tab contains your "Tech Entry Paths" code (unchanged), with the same helper kept above.

    # Your code repeats imports; they are already done above but re-importing is harmless.

    # ----- Your lines start (unchanged) -----
    import pandas as pd
    import re
    import matplotlib.pyplot as plt

    # NOTE: Your original code reads broad_sector.csv directly; we keep it unchanged.
    df = pd.read_csv('broad_sector.csv', on_bad_lines='skip')

    def to_period_str_quarter(s):
        if not s or pd.isna(s):
            return None
        s = str(s).strip()
        # Match pattern like "Q1 2017" (quarter + space + year)
        m = re.match(r'^[Qq]([1-4])\s+(\d{4})$', s)
        if m:
            q, y = m.groups()
            return f"{y}Q{q}"
        return None  # return None if not matched

    df['start_quarter_str'] = df['start_date'].apply(to_period_str_quarter)
    df['start_quarter'] = pd.PeriodIndex([p if p is not None else pd.NaT for p in df['start_quarter_str']], freq='Q')
    df = df.sort_values(['person_id', 'start_quarter'])

    df['next_sector'] = df.groupby('person_id')['broad_sector'].shift(-1)
    df['sector_change'] = (df['broad_sector'] != df['next_sector'])

    tech_entries = df[
        (df['next_sector'] == 'Tech / IT & Software') &
        (df['broad_sector'] != 'Tech / IT & Software')
    ]

    feeder_counts = tech_entries['broad_sector'].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10,5))
    feeder_counts.head(10).plot(kind='bar', color='teal')
    plt.title('🧭 Feeder Sectors → Tech / IT & Software')
    plt.ylabel('Number of People Entering Tech')
    plt.xlabel('Previous Broad Sector')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    print(feeder_counts.head(10))

    df['year'] = (
    df['start_quarter']
    .apply(lambda x: x.year if not pd.isna(x) else None)
)

# Filter to only 'Tech / IT & Software'
tech_entries = df[df['broad_sector'] == 'Tech / IT & Software'].copy()
tech_entries = tech_entries[
    (tech_entries['year'].notna()) &
    (tech_entries['year'] > 1970)
]

# Group by year
tech_year_counts = tech_entries.groupby('year').size()

# ---- Show chart or fallback message ----
if tech_year_counts.empty:
    print("⚠️ No valid Tech entries found with year > 1970. Check date format or 'broad_sector' spelling.")
else:
    plt.figure(figsize=(12, 6))
    tech_year_counts.plot(kind='bar', color='dodgerblue')
    plt.title("📈 First Entries into Tech Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of People Entering Tech")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
    # ----- Your lines end (unchanged) -----

st.markdown("---")
st.caption("Built for you — same visuals, same code paths; just wrapped with Streamlit tabs and render hooks.")
