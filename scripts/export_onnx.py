# https://mxnet.incubator.apache.org/tutorials/onnx/export_mxnet_to_onnx.html

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
    return key.split(":", 1)[1]

def fix_shape(symbol):
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
    # import pdb; pdb.set_trace()
    shapes = eval(shapes)

    converted_model_path = onnx_mxnet.export_model(symbol, params, shapes, np.float32, output, verbose=True)

    # Load onnx model
    model_proto = onnx.load_model(converted_model_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)

if __name__ == "__main__":
    main()