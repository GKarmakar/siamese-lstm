from flask import Flask
from argparse import ArgumentParser

import flask
import os

__DEFAULT_PORT = '6897'
__DEFAULT_HOST = '0.0.0.0'

app = Flask(__name__, template_folder='plots',
            root_path='.')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--port', dest='PORT', default=__DEFAULT_PORT)
    parser.add_argument('--host', dest='HOST', default=__DEFAULT_HOST)

    return parser.parse_args()


def main(args):
    try:
        app.run(host=args.HOST, port=args.PORT)
    finally:
        print()


@app.route('/')
def index():
    file_index = os.listdir('plots')
    file_index.remove('index.html')
    return flask.render_template('index.html', file_index=file_index)


@app.route('/<name>')
def render_plot(name):
    if name == 'favicon.ico':
        flask.abort(404)
    return flask.render_template(name)

if __name__ == '__main__':
    main(parse_arguments())
