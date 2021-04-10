from flask import Flask, jsonify, abort, make_response
import os
import responder

app = Flask(__name__)


@app.route("/<key>", methods=["GET"])
def main(key):
    try:
        return jsonify(responder.response(key))
    except:
        return jsonify(key+'?')


app.run()
