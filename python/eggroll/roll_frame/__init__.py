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
        elif isinstance(data, pa.RecordBatch):
            self._data = data
            self._schema = data.schema
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

    # TODO:0: implement numpy's empty interface
    @staticmethod
    def empty():
        return FrameBatch(data=None, schema=None)


class TensorBatch:
    META_SHAPE_KEY = bytes("eggroll.rollframe.tensor.shape", encoding="utf8")
    
    def __init__(self, data):
        if isinstance(data, FrameBatch):
            self._shape = [int(x) for x in str(data._schema.metadata[TensorBatch.META_SHAPE_KEY], encoding="utf8").split(",")]
            self._data = pa.Tensor.from_numpy(data._data.to_pandas().to_numpy().reshape(self._shape))
        elif isinstance(data, np.ndarray):
            self._data = pa.Tensor.from_numpy(data)
            self._shape = self._data.shape
        else:
            self._data = data
            self._shape = data.shape

    def to_numpy(self):
        return self._data.to_numpy()

    @staticmethod
    def from_numpy(np_array: np.ndarray):
        return TensorBatch(np_array)

    def to_frame(self):
        shape = bytes(",".join(str(x) for x in self._shape), encoding="utf8")
        return FrameBatch(pa.RecordBatch.from_arrays(arrays=[self.to_numpy().reshape(-1)],
                                                     names=['__data'],
                                                     metadata={TensorBatch.META_SHAPE_KEY: shape}))

    @staticmethod
    def from_frame(frame: FrameBatch):
        return TensorBatch(frame)


def create_functor(func_bin):
    try:
        return cloudpickle.loads(func_bin)
    except:
        return eggroll_pickle_loads(func_bin)
