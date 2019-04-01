from flask import Flask
from flask_restful import Resource, Api
from textgenrnn import textgenrnn
from flask import g

application = Flask(__name__)
api = Api(application)

def get_textgen():
    textgen = getattr(g, '_textgen', None)
    if textgen is None:
      textgen = g._textgen = textgenrnn(weights_path='showerthoughts_weights.hdf5', vocab_path='showerthoughts_vocab.json', config_path='showerthoughts_config.json')
    return textgen

class ShowerThoughts(Resource):
  def get(self):
    return 'Hello world'

  def post(self):
    output = get_textgen().generate(1, temperature=0.5, return_as_list=True, progress=False)
    return { 'response_type': 'in_channel', 'attachments':  [{ 'text': output[0] }] }

api.add_resource(ShowerThoughts, '/')

if __name__ == '__main__':
    application.debug = True
    application.run()
