import requests
import pandas as pd
import plotly.express as px

def fetch_earthquake_data(min_magnitude=4.5, days=7):
    """
    Fetch recent earthquakes from the USGS API.
    """
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=geojson&starttime=-{days}days&minmagnitude={min_magnitude}"
    )
    response = requests.get(url)
    data = response.json()

    features = data['features']
    records = []

    for feature in features:
        props = feature['properties']
        coords = feature['geometry']['coordinates']
        records.append({
            "Place": props['place'],
            "Magnitude": props['mag'],
            "Time": pd.to_datetime(props['time'], unit='ms'),
            "Longitude": coords[0],
            "Latitude": coords[1],
            "Depth_km": coords[2]
        })

    return pd.DataFrame(records)

def plot_earthquakes(df):
    """
    Plot earthquakes using Plotly scatter_geo.
    """
    fig = px.scatter_geo(
        df,
        lat='Latitude',
        lon='Longitude',
        color='Magnitude',
        size='Magnitude',
        hover_name='Place',
        hover_data=['Magnitude', 'Depth_km', 'Time'],
        title='Global Earthquakes (Magnitude > 4.5) from last 7 days',
        projection='natural earth',
        color_continuous_scale='Viridis'
    )
    fig.update_geos(showland=True, landcolor="LightGreen")
    fig.show()

# Main logic
if __name__ == "__main__":
    print("Fetching earthquake data...")
    df = fetch_earthquake_data(min_magnitude=4.5, days=7)
    print(f"Total Earthquakes: {len(df)}")
    print(df.head())

    print("Plotting earthquake map...")
    plot_earthquakes(df)
