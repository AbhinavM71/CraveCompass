from flask import Flask, render_template, request, redirect, flash
from sqlalchemy import create_engine, text
import os

app = Flask(
    __name__,
    static_folder='../frontend/static',
    template_folder='../frontend/templates'
)

app.secret_key = 'your_secret_key'

read_engine = create_engine('postgresql://zomato:zomato123@localhost:5433/zomato')   # SLAVE (READ)
write_engine = create_engine('postgresql://zomato:zomato123@localhost:5432/zomato')  # MASTER (WRITE)


@app.route('/')
def login():
    return render_template('index.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/home')
def home():
    return '<h1>Welcome to the Home Page</h1>'


@app.route('/signin', methods=['POST'])
def signin():
    email = request.form.get('email')
    password = request.form.get('password')

    with read_engine.connect() as conn:
        result = conn.execute(text("SELECT password FROM users WHERE email = :email"), {"email": email}).fetchone()

    if result is None:
        flash("Email not registered. Please sign up first.")
        return redirect('/')
    elif result[0] != password:
        flash("Incorrect password.")
        return redirect('/')
    else:
        return redirect('/home')


@app.route('/signup', methods=['POST'])
def create_account():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    # print(password)
    # print(confirm_password)
    if password != confirm_password:
        flash("Passwords do not match.")
        return redirect('/signup')

    with write_engine.begin() as conn:
        existing = conn.execute(text("SELECT 1 FROM users WHERE email = :email"), {"email": email}).fetchone()
        if existing:
            flash("Email already registered. Please sign in.")
            return redirect('/')
        conn.execute(text("""
            INSERT INTO users (email, name, password)
            VALUES (:email, :name, :password)
        """), {"email": email, "name": name, "password": password})

    flash("Account created successfully. Please sign in.")
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)