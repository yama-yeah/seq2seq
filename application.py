from flask import Flask, jsonify, abort, make_response
import os
import sys
from werkzeug.exceptions import Forbidden, HTTPException, NotFound, RequestTimeout, Unauthorized

PATH=os.path.abspath('') 
sys.path.append(PATH)

import responder

app = Flask(__name__)

@app.route("/")
def not_found():
    return jsonify("404")

@app.route("/<key>", methods=["GET"])
def main(key):
    try:
        return jsonify(responder.response(key))
    except:
        return jsonify(key+'?')
if __name__ == "__main__":
    app.run()
