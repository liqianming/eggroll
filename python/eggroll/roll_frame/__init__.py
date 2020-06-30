# -*- coding: utf-8 -*-
#  Copyright (c) 2019 - now, Eggroll Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Iterable

import cloudpickle
from eggroll.core.serdes.eggroll_serdes import PickleSerdes, \
    CloudPickleSerdes, EmptySerdes, eggroll_pickle_loads


class FrameBatch:
    def __init__(self, data, schema=None):
        if isinstance(data, FrameBatch):
            self._data = data._data
            self._schema = data._schema
        elif isinstance(data, bytes):
            context = pa.default_serialization_context()
            fb = FrameBatch(context.deserialize(data))
            self._data = fb._data
            self._schema = fb._schema
        elif isinstance(data, dict):
            fb = FrameBatch(pa.deserialize_components(data))
            self._data = fb._data
            self._schema = fb._schema
        elif isinstance(data, pd.DataFrame):
            fb = pa.RecordBatch.from_pandas(data)
            self._data = fb
            self._schema = fb.schema
        else:
            self._data = data
            self._schema = schema

    def to_pandas(self):
        return self._data.to_pandas()

    @staticmethod
    def from_pandas(obj):
        return FrameBatch(obj)

    @staticmethod
    def concat(frames: Iterable):
        frame = pd.concat(FrameBatch(f).to_pandas() for f in frames)
        return FrameBatch(frame)


class TensorBatch:
    META_SHAPE_KEY = bytes("eggroll.roll_frame.tensor.shape", encoding="utf8")
    
    def __init__(self, data):
        self._data = data

    def to_numpy(self):
        return self._data.to_numpy()

    @staticmethod
    def from_numpy(obj):
        return TensorBatch(pa.Tensor.from_numpy(obj))

    def to_frame(self):
        shape = bytes(",".join(str(x) for x in self._data.shape), encoding="utf8")
        return FrameBatch(pa.RecordBatch.from_arrays([self.to_numpy().reshape(-1)], names=['__f0'],
                                                     dict_metadata={TensorBatch.META_SHAPE_KEY: shape}))

    @staticmethod
    def from_frame(obj):
        shape = [int(x) for x in str(obj._data.schema.metadata[TensorBatch.META_SHAPE_KEY], encoding="utf8").split(",")]
        return FrameBatch(pa.Tensor.from_numpy(obj._data[0].to_numpy().reshape(shape)))


def create_functor(func_bin):
    try:
        return cloudpickle.loads(func_bin)
    except:
        return eggroll_pickle_loads(func_bin)
