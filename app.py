from flask import Flask, render_template, request, jsonify
from q_learning import QLearning

app = Flask(__name__)

# Create an instance of QLearning class
q_learning = QLearning()

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the grid page route
@app.route('/grid', methods=['GET', 'POST'])
def grid():
    if request.method == 'POST':
        n = int(request.form['n'])
        return render_template('grid.html', n=n)
    else:
        return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    grid = request.json['grid']
    start = request.json['start']
    end = request.json['end']
    blocks = request.json['blocks']

    q_learning.setup(grid, start, end, blocks)
    q_learning.train()

    path = q_learning.get_optimal_path()
    return jsonify({'path': path})

if __name__ == '__main__':
    app.run(debug=True)
