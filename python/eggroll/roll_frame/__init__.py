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


class FrameBatch:
    def __init__(self, data):
        self._data = data

    def to_pandas(self):
        return self._data.to_pandas()

    @staticmethod
    def from_pandas(obj):
        return FrameBatch(pa.RecordBatch.from_pandas(obj))


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
                                                     metadata={TensorBatch.META_SHAPE_KEY: shape}))

    @staticmethod
    def from_frame(obj):
        shape = [int(x) for x in str(obj._data.schema.metadata[TensorBatch.META_SHAPE_KEY], encoding="utf8").split(",")]
        return FrameBatch(pa.Tensor.from_numpy(obj._data[0].to_numpy().reshape(shape)))