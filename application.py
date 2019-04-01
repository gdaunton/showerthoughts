from flask import Flask
import tensorflow as tf
from flask_restful import Resource, Api
from textgenrnn import textgenrnn

application = Flask(__name__)
api = Api(application)

class ShowerThoughts(Resource):
  def get(self):
    return 'Hello world'

  def post(self):
    output = textgen.generate(1, temperature=0.5, return_as_list=True, progress=False)
    return { 'response_type': 'in_channel', 'attachments':  [{ 'text': output[0] }] }

api.add_resource(ShowerThoughts, '/')

if __name__ == '__main__':
  global textgen
  application.debug = True
  textgen = textgenrnn(weights_path='showerthoughts_weights.hdf5', vocab_path='showerthoughts_vocab.json', config_path='showerthoughts_config.json')
  textgen.generate()
  application.run()
