from flask import Flask, render_template, request, redirect, flash, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from tensorflow.keras.optimizers.experimental import AdamW
import math
from math import ceil
# — load your TF model —
model = tf.keras.models.load_model(
    r"C:\Users\abhin\OneDrive - iitr.ac.in\OneDrive\Desktop\CraveCompass\CraveCompass\food_classifier\Food_Vision_Model.h5",
    compile=False
)
model.compile(
    optimizer=AdamW(weight_decay=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# — class names & cuisine map —
class_names = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
    "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
    # … all the way through …
    "tiramisu", "tuna tartare", "waffles"
]

dish_to_cuisine = {
    "bibimbap": "Korean",
    "sushi": "Japanese",
    "ramen": "Japanese",
    "takoyaki": "Japanese",
    "pad thai": "Thai",
    "chicken curry": "Indian",
    "samosa": "Indian",
    "pizza": "Italian",
    "spaghetti bolognese": "Italian",
    "lasagna": "Italian",
    "tacos": "Mexican",
    "guacamole": "Mexican",
    "apple pie": "American",
    "baby back ribs": "American",
    "poutine": "Canadian",
    "baklava": "Middle Eastern",
    "falafel": "Middle Eastern",
    # … map the rest …
}

app = Flask(
    __name__,
    static_folder='../frontend/static',
    template_folder='../frontend/templates'
)
app.secret_key = 'your_secret_key'

# — DB engines —
read_engine  = create_engine('postgresql://zomato:zomato123@localhost:5433/zomato')
write_engine = create_engine('postgresql://zomato:zomato123@localhost:5432/zomato')


@app.route('/')
def login():
    return render_template('index.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/signin', methods=['POST'])
def signin():
    email = request.form.get('email')
    password = request.form.get('password')

    with read_engine.connect() as conn:
        result = conn.execute(
            text("SELECT password FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()

    if result is None:
        flash("Email not registered. Please sign up first.")
        return redirect('/')
    elif result[0] != password:
        flash("Incorrect password.")
        return redirect('/')
    else:
        return redirect('/home')



@app.route('/search_by_text', methods=['GET', 'POST'])
def search_by_text():
    # ── Pagination setup ───────────────────────────────────────
    page     = request.args.get('page', 1, type=int)
    per_page = 10
    offset   = (page - 1) * per_page

    # ── Get the query string from POST (new search) or GET (pagination) ──
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
    else:
        query = request.args.get('query', '').strip()

    if not query:
        flash("Search query cannot be empty.", "warning")
        return redirect('/home')

    # ── Try exact ID match ─────────────────────────────────────
    try:
        query_id = int(query)
    except ValueError:
        query_id = -1

    # ── Base WHERE clause & params ─────────────────────────────
    where_clause = """
      (r.restaurant_name ILIKE :q OR
       r.cuisines         ILIKE :q OR
       r.locality         ILIKE :q OR
       r.city             ILIKE :q OR
       r.restaurant_id = :qid)
    """
    params = {
        "q": f"%{query}%",
        "qid": query_id,
        "limit": per_page,
        "offset": offset
    }

    # ── Fetch paginated rows ────────────────────────────────────
    sql = f"""
    SELECT
      r.*,
      ST_Y(r.geom::geometry) AS latitude,
      ST_X(r.geom::geometry) AS longitude
    FROM restaurants AS r
    WHERE {where_clause}
    ORDER BY r.restaurant_id
    LIMIT :limit OFFSET :offset
    """
    with read_engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    # ── Total count for calculating total_pages ─────────────────
    count_sql = f"SELECT COUNT(*) FROM restaurants AS r WHERE {where_clause}"
    with read_engine.connect() as conn:
        total = conn.execute(text(count_sql), {
            "q": params["q"],
            "qid": params["qid"]
        }).scalar()

    total_pages = ceil(total / per_page)

    # ── Map to dicts ────────────────────────────────────────────
    restaurants = [{
        "id": row.restaurant_id,
        "name": row.restaurant_name,
        "country_code": row.country_code,
        "country": row.country,
        "city": row.city,
        "address": row.address,
        "locality": row.locality,
        "locality_verbose": row.locality_verbose,
        "longitude": row.longitude,
        "latitude": row.latitude,
        "cuisines": row.cuisines,
        "average_cost_for_two": row.average_cost_for_two,
        "currency": row.currency,
        "has_table_booking": row.has_table_booking,
        "has_online_delivery": row.has_online_delivery,
        "is_delivering_now": row.is_delivering_now,
        "switch_to_order_menu": row.switch_to_order_menu,
        "price_range": row.price_range,
        "aggregate_rating": row.aggregate_rating,
        "rating_color": row.rating_color,
        "rating_text": row.rating_text,
        "votes": row.votes,
        "lat": row.latitude,
        "lon": row.longitude,
    } for row in rows]

    # ── Render with pagination vars ─────────────────────────────
    return render_template(
        'home.html',
        restaurants=restaurants,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        query=query
    )



@app.route('/signup', methods=['POST'])
def create_account():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    if password != confirm_password:
        flash("Passwords do not match.")
        return redirect('/signup')

    with write_engine.begin() as conn:
        existing = conn.execute(
            text("SELECT 1 FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
        if existing:
            flash("Email already registered. Please sign in.")
            return redirect('/')
        conn.execute(text("""
            INSERT INTO users (email, name, password)
            VALUES (:email, :name, :password)
        """), {"email": email, "name": name, "password": password})

    flash("Account created successfully. Please sign in.")
    return redirect('/')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict_cuisine', methods=['POST'])
def predict_cuisine():
    f = request.files.get('image')
    if not f:
        flash("Select an image.", "warning")
        return redirect(url_for('home'))

    img = Image.open(f.stream).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    preds = model.predict(np.expand_dims(arr, 0))
    idx = int(np.argmax(preds, axis=1)[0])
    dish = class_names[idx]
    cuisine = dish_to_cuisine.get(dish, "Unknown")

    return render_template(
        'home.html',
        predicted_dish=dish,
        predicted_cuisine=cuisine
    )


@app.route('/restaurants/<int:rid>')
def restaurant_detail(rid):
    with read_engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM restaurants WHERE restaurant_id = :id"),
            {"id": rid}
        ).fetchone()

    if row is None:
        flash("Restaurant not found.", "warning")
        return redirect('/home')

    return render_template("restaurant_detail.html", r=row)


@app.route('/search_restaurants', methods=['GET','POST'])
def search_restaurants():
    # pagination
    page     = request.args.get('page', 1, type=int)
    per_page = 10
    offset   = (page - 1) * per_page

    # read filters from POST or GET
    if request.method == 'POST':
        lon        = float(request.form.get('longitude', 0))
        lat        = float(request.form.get('latitude', 0))
        cuisine    = request.form.get('cuisine', '').strip()
        country    = request.form.get('country', '').strip()
        cost_limit = float(request.form.get('cost_for_two', 0))
    else:
        lon        = float(request.args.get('longitude', 0))
        lat        = float(request.args.get('latitude', 0))
        cuisine    = request.args.get('cuisine', '').strip()
        country    = request.args.get('country', '').strip()
        cost_limit = float(request.args.get('cost_for_two', 0))

    sql = """
    SELECT
      r.*,
      ST_Y(r.geom::geometry) AS latitude,
      ST_X(r.geom::geometry) AS longitude
    FROM restaurants AS r
    WHERE ST_DWithin(
      r.geom::geography,
      ST_SetSRID(ST_MakePoint(:lon, :lat),4326)::geography,
      3000
    )
    """
    params = {"lon": lon, "lat": lat}

    if cuisine:
        sql += " AND r.cuisines ILIKE '%' || :cuisine || '%'"
        params["cuisine"] = cuisine
    if country:
        sql += " AND r.country = :country"
        params["country"] = country
    if cost_limit > 0:
        sql += " AND r.average_cost_for_two <= :cost_limit"
        params["cost_limit"] = cost_limit

    sql += " ORDER BY r.restaurant_id LIMIT :limit OFFSET :offset"
    params["limit"]  = per_page
    params["offset"] = offset

    with read_engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    # total count for pagination
    count_sql = """
    SELECT COUNT(*) FROM restaurants AS r
    WHERE ST_DWithin(
      r.geom::geography,
      ST_SetSRID(ST_MakePoint(:lon, :lat),4326)::geography,
      3000
    )
    """
    count_params = {"lon": lon, "lat": lat}
    if cuisine:
        count_sql    += " AND r.cuisines ILIKE '%' || :cuisine || '%'"
        count_params["cuisine"] = cuisine
    if country:
        count_sql    += " AND r.country = :country"
        count_params["country"] = country
    if cost_limit > 0:
        count_sql    += " AND r.average_cost_for_two <= :cost_limit"
        count_params["cost_limit"] = cost_limit

    with read_engine.connect() as conn:
        total = conn.execute(text(count_sql), count_params).scalar()

    total_pages = math.ceil(total / per_page)

    restaurants = [{
        "id":                   row.restaurant_id,
        "name":                 row.restaurant_name,
        "country_code":         row.country_code,
        "country":              row.country,
        "city":                 row.city,
        "address":              row.address,
        "locality":             row.locality,
        "locality_verbose":     row.locality_verbose,
        "longitude":            row.longitude,
        "latitude":             row.latitude,
        "cuisines":             row.cuisines,
        "average_cost_for_two": row.average_cost_for_two,
        "currency":             row.currency,
        "has_table_booking":    row.has_table_booking,
        "has_online_delivery":  row.has_online_delivery,
        "is_delivering_now":    row.is_delivering_now,
        "switch_to_order_menu": row.switch_to_order_menu,
        "price_range":          row.price_range,
        "aggregate_rating":     row.aggregate_rating,
        "rating_color":         row.rating_color,
        "rating_text":          row.rating_text,
        "votes":                row.votes,
        "lat":                  row.latitude,
        "lon":                  row.longitude,
    } for row in rows]

    return render_template(
        'home.html',
        restaurants=restaurants,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        longitude=lon,
        latitude=lat,
        cuisine=cuisine,
        country=country,
        cost_for_two=cost_limit
    )


if __name__ == '__main__':
    app.run(debug=True)
