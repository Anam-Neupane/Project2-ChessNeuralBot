# The HTTP bridge between the game and the neural network. 

import os 
import sys

# Make src/ importable no matter the working directory. 
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import chess 
import flask 

from search import pick_move

app = flask.Flask(__name__)

@app.route("/health", methods = ["GET"])
def health(): 
    """The Cpp side polls this until the server is ready to serve."""
    return {"status": "ok"}

@app.route("/get_move", methods=["POST"])
def get_move():
    # Read the JSON body. silent=True means "return None instead of erroring" 
    # if the both isn't valid JSON. 
    data = flask.request.get_json(silent=True)
    if not data or "moves" not in data: 
        return flask.jsonify({"error": "missing 'moves' in body"}), 400 
     
    board = chess.Board()
    for uci in data["moves"]: 
        try: 
            board.push_uci(uci)
        except ValueError: 
            return flask.jsonify({"error": f"illegal or unknown move: {uci}"}), 400 
    
    if board.is_game_over(): 
        return flask.jsonify({"move": None, "game_over": True})
    
    move = pick_move(board) 
    return flask.jsonify({"move": move.uci() if move else None})

if __name__ == "__main__":
    from waitress import serve

    serve(app, host="127.0.0.1", port=5000)