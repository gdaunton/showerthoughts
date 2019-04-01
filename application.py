from flask import Flask
import tensorflow as tf
from flask_restful import Resource, Api
from textgenrnn import textgenrnn
import threading

application = Flask(__name__)
api = Api(application)

global textgen, output
textgen = textgenrnn(weights_path='showerthoughts_weights.hdf5', vocab_path='showerthoughts_vocab.json', config_path='showerthoughts_config.json')
output = textgen.generate(5, temperature=0.5, return_as_list=True, progress=False)

def calcNewItems():
  global output
  print('generating new thoughts')
  output += textgen.generate(5, temperature=0.5, return_as_list=True, progress=False)

class ShowerThoughts(Resource):
  def get(self):
    return 'Hello world'

  def post(self):
    global output
    if(len(output) <= 2):
      thr = threading.Thread(target=calcNewItems)
      thr.start()
    if(len(output) >= 1):
      return { 'response_type': 'in_channel', 'attachments':  [{ 'text': output.pop() }] }
    return { 'response_type': 'in_channel', 'attachments':  [{ 'text': 'try again' }] }

api.add_resource(ShowerThoughts, '/')

if __name__ == '__main__':
  application.debug = True
  application.run(port=12345)
