"""Convert as saved model to onnx format.

The script is made after the following tutorial:
# https://mxnet.incubator.apache.org/tutorials/onnx/export_mxnet_to_onnx.html

An example to call this script.
```
python export_onnx.py
  --symbol epoch_18_acc_0.80_loss_0.91.json 
  --params epoch_18_acc_0.80_loss_0.91.params 
  --shapes "[(1,3,32,32),(1,)]" 
  --output epoch_18_acc_0.80_loss_0.91.onnx
```
"""

import os
import re
import json

import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
from onnx import checker
import onnx
import click

def strip_prefix(key):
    """Remove "arg" or "aux" prefix for all the keys in the params file.

    The prefix are kept in the params for compatibility reason. To use the saved
    model in mxnet, see https://gist.github.com/pierric/b4da8783755de763bff13aea5f5e900d
    
    Arguments:
        key {str} -- a key in the params dict
    
    Returns:
        str -- key without prefix
    """

    return key.split(":", 1)[1]

def fix_shape(symbol):
    """fix up the shapes in the attributes of nodes.

    mxnet allows shape in the form of list, however the official library to export
    model to onnx only accepts shape in the form of tuple. Here we fix up all shapes
    in the model.
    
    Arguments:
        symbol {dict} -- symbol's json
    """

    list_re = re.compile('\[([0-9L|,| ]+)\]')
    for op in symbol["nodes"]:
        attrs = op.get("attrs", {})
        for key in attrs:
            if isinstance(attrs[key], str):
                match = re.match(list_re, attrs[key])
                if match:
                    attrs[key] = "(" + match.group(1) + ")"


@click.command()
@click.option("--symbol", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--params", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(exists=False, dir_okay=False))
@click.option("--shapes", required=True, type=str)
def main(symbol, params, output, shapes):

    with open(symbol, "r") as fp:
        symbol_def = json.load(fp)
        fix_shape(symbol_def)
        with open("a.json", 'w') as o:
            json.dump(symbol_def, o)
        symbol = mx.sym.load_json(json.dumps(symbol_def))

    params = mx.nd.load(params)
    params = {strip_prefix(key): val for key, val in params.items()}
    shapes = eval(shapes)

    converted_model_path = onnx_mxnet.export_model(symbol, params, shapes, np.float32, output, verbose=True)

    # Load onnx model
    model_proto = onnx.load_model(converted_model_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)

if __name__ == "__main__":
    main()