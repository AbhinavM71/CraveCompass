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
# Replace your model loading with this
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
class_names = ['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheese_plate',
 'cheesecake',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles'] 

def create_food_classifier():
    # Use a standard pre-trained model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add a custom classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    
    # Create the model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

# Comment out your existing model loading
# model = tf.keras.models.load_model(...)
# Use this instead
model = create_food_classifier()
print("Model loaded successfully")
print(f"Model summary: {model.summary()}")
# model.compile(
#     optimizer=AdamW(weight_decay=1e-5),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# — class names & cuisine map —

dish_to_cuisine = {
    # American
    "apple_pie": "American",
    "baby_back_ribs": "American",
    "beef_carpaccio": "American",
    "beef_tartare": "American",
    "beet_salad": "American",
    "breakfast_burrito": "American",
    "caesar_salad": "American",
    "carrot_cake": "American",
    "cheese_plate": "American",
    "chicken_wings": "American",
    "chocolate_cake": "American",
    "club_sandwich": "American",
    "crab_cakes": "American",
    "cup_cakes": "American",
    "deviled_eggs": "American",
    "donuts": "American",
    "eggs_benedict": "American",
    "filet_mignon": "American",
    "french_fries": "American",
    "french_toast": "American",
    "fried_calamari": "American",
    "frozen_yogurt": "American",
    "grilled_cheese_sandwich": "American",
    "grilled_salmon": "American",
    "hamburger": "American",
    "hot_dog": "American",
    "huevos_rancheros": "American",
    "ice_cream": "American",
    "lobster_roll_sandwich": "American",
    "macaroni_and_cheese": "American",
    "onion_rings": "American",
    "pancakes": "American",
    "pork_chop": "American",
    "prime_rib": "American",
    "pulled_pork_sandwich": "American",
    "red_velvet_cake": "American",
    "shrimp_and_grits": "American",
    "steak": "American",
    "strawberry_shortcake": "American",
    "waffles": "American",
    
    # French
    "beignets": "French",
    "bruschetta": "French",
    "chocolate_mousse": "French",
    "creme_brulee": "French",
    "croque_madame": "French",
    "escargots": "French",
    "foie_gras": "French",
    "french_onion_soup": "French",
    "macarons": "French",
    "panna_cotta": "French",
    
    # Italian
    "cannoli": "Italian",
    "caprese_salad": "Italian",
    "cheese_plate": "Italian",
    "cheesecake": "Italian",
    "garlic_bread": "Italian",
    "gnocchi": "Italian",
    "lasagna": "Italian",
    "lobster_bisque": "Italian",
    "pizza": "Italian",
    "ravioli": "Italian",
    "risotto": "Italian",
    "spaghetti_bolognese": "Italian",
    "spaghetti_carbonara": "Italian",
    "tiramisu": "Italian",
    
    # Mexican
    "chicken_quesadilla": "Mexican",
    "churros": "Mexican",
    "guacamole": "Mexican",
    "nachos": "Mexican",
    "tacos": "Mexican",
    
    # Asian (Various)
    "bibimbap": "Korean",
    "chicken_curry": "Indian",
    "dumplings": "Chinese",
    "edamame": "Japanese",
    "fried_rice": "Chinese",
    "gyoza": "Japanese",
    "hot_and_sour_soup": "Chinese",
    "miso_soup": "Japanese",
    "pad_thai": "Thai",
    "paella": "Spanish",
    "peking_duck": "Chinese",
    "pho": "Vietnamese",
    "ramen": "Japanese",
    "samosa": "Indian",
    "sashimi": "Japanese",
    "seaweed_salad": "Japanese",
    "spring_rolls": "Chinese",
    "sushi": "Japanese",
    "takoyaki": "Japanese",
    
    # Mediterranean/Middle Eastern
    "baklava": "Middle Eastern",
    "falafel": "Middle Eastern",
    "greek_salad": "Greek",
    "hummus": "Middle Eastern",
    
    # Seafood (Various Cuisines)
    "clam_chowder": "American",
    "fish_and_chips": "British",
    "mussels": "French",
    "oysters": "French",
    "scallops": "French",
    "tuna_tartare": "Japanese",
    
    # Other
    "bread_pudding": "British",
    "omelette": "French",
    "poutine": "Canadian"
}

# Create a list of unique cuisines
cuisines = sorted(list(set(dish_to_cuisine.values())))

print("Unique cuisines:", cuisines) 

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


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tempfile
import os




@app.route('/predict_cuisine', methods=['POST'])
def predict_cuisine():
    f = request.files.get('image')
    if not f:
        flash("Select an image.", "warning")
        return redirect(url_for('home'))
    
    base_dir = r"C:\Users\abhin\OneDrive - iitr.ac.in\OneDrive\Desktop\CraveCompass\CraveCompass\frontend\static\image_docs"
    temp_path = os.path.join(base_dir, "temp_food_image.jpg")
    
    try:
        # Save the file
        f.save(temp_path)
        print(f"File saved to {temp_path}")
        
        # Use Keras preprocessing
        img = load_img(temp_path, target_size=(224, 224))
        img_array = img_to_array(img)
        
        # Print original image stats
        print(f"Original image shape: {img_array.shape}")
        print(f"Original min value: {np.min(img_array)}, max value: {np.max(img_array)}")
        
        # Try simple [0,1] normalization
        img_array_normalized = img_array / 255.0
        img_array_normalized = np.expand_dims(img_array_normalized, axis=0)
        
        # Print stats after normalization
        print(f"Normalized min value: {np.min(img_array_normalized)}, max value: {np.max(img_array_normalized)}")
        
        # Make prediction with normalized input
        print("Predicting with normalized input...")
        preds = model.predict(img_array_normalized)
        
        # Check for NaN values
        if np.isnan(np.sum(preds)):
            print("Warning: NaN predictions with normalized input - trying different preprocessing")
            
            # Try with MobileNetV2 preprocessing
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
            img_array_mobilenet = mobilenet_preprocess(img_array.copy())
            img_array_mobilenet = np.expand_dims(img_array_mobilenet, axis=0)
            
            # Print stats after MobileNet preprocessing
            print(f"MobileNet preprocessed min: {np.min(img_array_mobilenet)}, max: {np.max(img_array_mobilenet)}")
            
            # Try prediction with MobileNet preprocessing
            preds = model.predict(img_array_mobilenet)
            
            if np.isnan(np.sum(preds)):
                print("Warning: Still getting NaN predictions - using fallback")
                dish = "pizza"
                cuisine = "Italian" 
                confidence = 0
            else:
                idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][idx])
                dish = class_names[idx]
                cuisine = dish_to_cuisine.get(dish, "Unknown")
        else:
            idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(preds[0][idx])
            print(confidence)
            dish = class_names[idx]
            cuisine = dish_to_cuisine.get(dish, "Unknown")
            
    except Exception as e:
        print(f"Error in predict_cuisine: {str(e)}")
        dish = "error"
        cuisine = "Unknown"
        confidence = 0
    
    return render_template(
        'home.html',
        predicted_dish=dish,
        predicted_cuisine=cuisine,
        confidence=f"{confidence*100:.2f}%"
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
