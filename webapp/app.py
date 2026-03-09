"""
Flask Web Application - Dashboard básico
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    """Home page - Dashboard placeholder"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check"""
    return {'status': 'ok'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
