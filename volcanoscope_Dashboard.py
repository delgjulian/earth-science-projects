# volcanoscope_dashboard.py

# VolcanoScope: Global Monitoring & Visualization of Active Volcanoes# volcanoscope_monitor.py

"""
VolcanoScope - Global Active Volcano Visualization
This program uses a statistically significant sample of active volcanoes inspired by the Smithsonian GVP database
and displays them on an interactive world map.
"""

import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random

def load_statistical_volcano_data():
    """
    Loads a statistically representative sample of active volcanoes across regions with mock last activity dates.
    """
    data = [
        {"Volcano": "Mount Etna", "Country": "Italy", "Latitude": 37.748, "Longitude": 14.999, "Elevation_m": 3329, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Kilauea", "Country": "USA (Hawaii)", "Latitude": 19.421, "Longitude": -155.287, "Elevation_m": 1247, "Type": "Shield volcano", "Status": "Erupting"},
        {"Volcano": "Popocatépetl", "Country": "Mexico", "Latitude": 19.023, "Longitude": -98.622, "Elevation_m": 5426, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Taal", "Country": "Philippines", "Latitude": 14.002, "Longitude": 120.993, "Elevation_m": 311, "Type": "Caldera", "Status": "Restless"},
        {"Volcano": "Nevado del Ruiz", "Country": "Colombia", "Latitude": 4.892, "Longitude": -75.324, "Elevation_m": 5321, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Mount Fuji", "Country": "Japan", "Latitude": 35.3606, "Longitude": 138.7274, "Elevation_m": 3776, "Type": "Stratovolcano", "Status": "Dormant"},
        {"Volcano": "Sakurajima", "Country": "Japan", "Latitude": 31.585, "Longitude": 130.657, "Elevation_m": 1117, "Type": "Stratovolcano", "Status": "Erupting"},
        {"Volcano": "Villarrica", "Country": "Chile", "Latitude": -39.42, "Longitude": -71.93, "Elevation_m": 2847, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Nyiragongo", "Country": "DR Congo", "Latitude": -1.52, "Longitude": 29.25, "Elevation_m": 3470, "Type": "Stratovolcano", "Status": "Restless"},
        {"Volcano": "Mount Erebus", "Country": "Antarctica", "Latitude": -77.53, "Longitude": 167.17, "Elevation_m": 3794, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Shiveluch", "Country": "Russia", "Latitude": 56.653, "Longitude": 161.360, "Elevation_m": 3283, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Fuego", "Country": "Guatemala", "Latitude": 14.473, "Longitude": -90.880, "Elevation_m": 3763, "Type": "Stratovolcano", "Status": "Erupting"},
        {"Volcano": "White Island", "Country": "New Zealand", "Latitude": -37.52, "Longitude": 177.18, "Elevation_m": 321, "Type": "Stratovolcano", "Status": "Restless"},
        {"Volcano": "Mount Merapi", "Country": "Indonesia", "Latitude": -7.5407, "Longitude": 110.4462, "Elevation_m": 2930, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Soufrière Hills", "Country": "Montserrat", "Latitude": 16.72, "Longitude": -62.18, "Elevation_m": 1050, "Type": "Stratovolcano", "Status": "Restless"},
        {"Volcano": "Mount St. Helens", "Country": "USA", "Latitude": 46.1912, "Longitude": -122.1944, "Elevation_m": 2549, "Type": "Stratovolcano", "Status": "Dormant"},
        {"Volcano": "Mount Nyamuragira", "Country": "DR Congo", "Latitude": -1.41, "Longitude": 29.20, "Elevation_m": 3058, "Type": "Shield volcano", "Status": "Erupting"},
        {"Volcano": "Cumbre Vieja", "Country": "Spain (Canary Islands)", "Latitude": 28.57, "Longitude": -17.83, "Elevation_m": 1949, "Type": "Stratovolcano", "Status": "Restless"},
        {"Volcano": "Mount Cleveland", "Country": "USA (Alaska)", "Latitude": 52.821, "Longitude": -169.944, "Elevation_m": 1730, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Cotopaxi", "Country": "Ecuador", "Latitude": -0.677, "Longitude": -78.437, "Elevation_m": 5897, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Mauna Loa", "Country": "USA (Hawaii)", "Latitude": 19.475, "Longitude": -155.608, "Elevation_m": 4169, "Type": "Shield volcano", "Status": "Dormant"},
        {"Volcano": "Mount Agung", "Country": "Indonesia", "Latitude": -8.343, "Longitude": 115.508, "Elevation_m": 3031, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Mount Taranaki", "Country": "New Zealand", "Latitude": -39.296, "Longitude": 174.063, "Elevation_m": 2518, "Type": "Stratovolcano", "Status": "Dormant"},
        {"Volcano": "Sangay", "Country": "Ecuador", "Latitude": -2.005, "Longitude": -78.341, "Elevation_m": 5230, "Type": "Stratovolcano", "Status": "Erupting"},
        {"Volcano": "Mount Rinjani", "Country": "Indonesia", "Latitude": -8.42, "Longitude": 116.47, "Elevation_m": 3726, "Type": "Stratovolcano", "Status": "Restless"},
        {"Volcano": "Ol Doinyo Lengai", "Country": "Tanzania", "Latitude": -2.764, "Longitude": 35.914, "Elevation_m": 2962, "Type": "Stratovolcano", "Status": "Active"},
        {"Volcano": "Mount Cameroon", "Country": "Cameroon", "Latitude": 4.203, "Longitude": 9.170, "Elevation_m": 4040, "Type": "Stratovolcano", "Status": "Erupting"},
        {"Volcano": "Mount Baker", "Country": "USA", "Latitude": 48.777, "Longitude": -121.813, "Elevation_m": 3286, "Type": "Stratovolcano", "Status": "Dormant"},
        {"Volcano": "Mount Pelée", "Country": "Martinique", "Latitude": 14.809, "Longitude": -61.165, "Elevation_m": 1397, "Type": "Stratovolcano", "Status": "Restless"},
        {"Volcano": "Mount Ararat", "Country": "Turkey", "Latitude": 39.702, "Longitude": 44.298, "Elevation_m": 5137, "Type": "Stratovolcano", "Status": "Dormant"},
        {"Volcano": "Heard Island", "Country": "Australia", "Latitude": -53.106, "Longitude": 73.504, "Elevation_m": 2745, "Type": "Stratovolcano", "Status": "Active"}
    ]

    # Create DataFrame
    df = pd.DataFrame(data)

    # Generate random last activity dates within the past 2 years
    today = datetime.utcnow()
    df["LastActivity"] = [today - timedelta(days=random.randint(0, 730)) for _ in range(len(df))]
    df["DaysAgo"] = (today - df["LastActivity"]).dt.days
    return df

def plot_active_volcanoes(df):
    """
    Plots volcano locations with elevation and activity status and time dimension.
    """
    fig = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        text='Volcano',
        color='Status',
        size='Elevation_m',
        hover_name='Volcano',
        hover_data={
            'Country': True,
            'Type': True,
            'Elevation_m': True,
            'LastActivity': True,
            'DaysAgo': True
        },
        title='Active Volcanoes with Last Known Activity (Sample - Smithsonian GVP Inspired)',
        projection='natural earth',
        template='plotly_white'
    )
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(235, 235, 235)",
            showocean=True,
            oceancolor="rgb(210, 230, 255)",
            showcountries=True,
            countrycolor="rgb(190, 190, 190)"
        )
    )
    fig.show()

# Main logic
if __name__ == "__main__":
    df = load_statistical_volcano_data()
    plot_active_volcanoes(df)
