from flask import Flask, jsonify, abort, make_response
import os
import sys
from werkzeug.exceptions import Forbidden, HTTPException, NotFound, RequestTimeout, Unauthorized

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

@app.errorhandler(NotFound)
def page_not_found_handler(e: HTTPException):
    return jsonify('404')


@app.errorhandler(Unauthorized)
def unauthorized_handler(e: HTTPException):
    return jsonify('401')


@app.errorhandler(Forbidden)
def forbidden_handler(e: HTTPException):
    return jsonify('403')


@app.errorhandler(RequestTimeout)
def request_timeout_handler(e: HTTPException):
    return render_template('408.html'), 408

app.run()
