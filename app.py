import os
from flask import Flask, render_template, request, redirect, url_for, session
import subprocess

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Dummy database for registered users (replace with your actual database)
registered_users = {
    'admin': 'password123'
}

# Function to authenticate users
def authenticate(username, password):
    if username in registered_users and registered_users[username] == password:
        return True
    return False

# Function to register new users
def register(username, password):
    if username not in registered_users:
        registered_users[username] = password
        return True
    return False

# Flask route for login page
@app.route("/")
def login():
    return render_template("login.html")

# Flask route for login form submission
@app.route("/login", methods=["POST"])
def login_post():
    username = request.form.get("username")
    password = request.form.get("password")

    if authenticate(username, password):
        # Set session variable to mark user as logged in
        session['username'] = username
        # Redirect to the Streamlit page
        return redirect(url_for("streamlit"))
    else:
        return render_template("login.html", message="Invalid username or password")

# Flask route for registration page
@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if register(username, password):
            return redirect(url_for("login"))
        else:
            return render_template("register.html", message="Registration failed. Username already exists.")
    return render_template("register.html")

# Flask route for Streamlit page
@app.route("/streamlit")
def streamlit():
    if 'username' not in session:
        return redirect(url_for("login"))
    
    # Execute Streamlit application
    streamlit_process = subprocess.Popen(['python', 'main.py'])
    return render_template("streamlit.html")

if __name__ == "__main__":
    app.run(debug=True)
