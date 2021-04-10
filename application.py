from flask import Flask, jsonify, abort, make_response
import os
import sys

PATH=os.path.abspath('') 
sys.path.append(PATH)

import responder

app = Flask(__name__)


@app.route("/<key>", methods=["GET"])
def main(key):
    try:
        return jsonify(responder.response(key))
    except:
        return jsonify(key+'?')


app.run()
