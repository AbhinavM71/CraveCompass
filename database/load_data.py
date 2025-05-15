import pandas as pd
from sqlalchemy import create_engine, text

# Paths to data files
CSV_PATH     = r"C:\Users\abhin\OneDrive - iitr.ac.in\OneDrive\Desktop\CraveCompass\CraveCompass\zomato-restaurants-data\zomato.csv"
COUNTRY_PATH = r"C:\Users\abhin\OneDrive - iitr.ac.in\OneDrive\Desktop\CraveCompass\CraveCompass\zomato-restaurants-data\Country-Code.xlsx"

# Columns to load from CSV
COLUMNS = [
    "Restaurant ID", "Restaurant Name", "Country Code", "City", "Address", "Locality",
    "Locality Verbose", "Longitude", "Latitude", "Cuisines", "Average Cost for two",
    "Currency", "Has Table booking", "Has Online delivery", "Is delivering now",
    "Switch to order menu", "Price range", "Aggregate rating", "Rating color",
    "Rating text", "Votes"
]

# Load and clean data
df = pd.read_csv(CSV_PATH, usecols=COLUMNS, encoding="latin-1")
df_country = pd.read_excel(COUNTRY_PATH)
df = df.merge(df_country, on="Country Code", how="left")
df.columns = [
    c.strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    for c in df.columns
]
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"],  errors="coerce")
df["average_cost_for_two"] = pd.to_numeric(df["average_cost_for_two"], errors="coerce")
df["aggregate_rating"]     = pd.to_numeric(df["aggregate_rating"], errors="coerce")
df["votes"]                = pd.to_numeric(df["votes"], errors="coerce")

# Connect to Postgres
engine = create_engine("postgresql://zomato:zomato123@localhost:5432/zomato")

with engine.begin() as conn:
    # Enable required extensions
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))

    # Drop and recreate restaurants table
    conn.execute(text("DROP TABLE IF EXISTS restaurants;"))
    conn.execute(text("""
    CREATE TABLE restaurants (
        restaurant_id         INTEGER PRIMARY KEY,
        restaurant_name       TEXT,
        country_code          INTEGER,
        country               TEXT,
        city                  TEXT,
        address               TEXT,
        locality              TEXT,
        locality_verbose      TEXT,
        longitude             DOUBLE PRECISION,
        latitude              DOUBLE PRECISION,
        cuisines              TEXT,
        average_cost_for_two  DOUBLE PRECISION,
        currency              TEXT,
        has_table_booking     BOOLEAN,
        has_online_delivery   BOOLEAN,
        is_delivering_now     BOOLEAN,
        switch_to_order_menu  BOOLEAN,
        price_range           INTEGER,
        aggregate_rating      DOUBLE PRECISION,
        rating_color          TEXT,
        rating_text           TEXT,
        votes                 INTEGER,
        geom                  geography(Point,4326)
    );
    """))

    # 🔥 Create the Users table (new)
    conn.execute(text("DROP TABLE IF EXISTS users;"))
    conn.execute(text("""
    CREATE TABLE users (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        password TEXT NOT NULL
    );
    """))

# Load data into restaurants table
df.to_sql("restaurants", engine, index=False, if_exists="append")

# Update geom and add spatial index
with engine.begin() as conn:
    conn.execute(text("""
        UPDATE restaurants
           SET geom = ST_SetSRID(
               ST_MakePoint(longitude, latitude),
               4326
           )::geography;
    """))
    conn.execute(text("CREATE INDEX ON restaurants USING GIST (geom);"))

print("✅ Loaded", len(df), "restaurants into Postgres, and created Users table.")
