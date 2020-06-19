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
import unittest
import pyarrow as pa
import numpy as np
import pandas as pd

from eggroll.roll_frame import FrameBatch, TensorBatch
from eggroll.roll_frame.frame_store import create_frame_adapter


class TestStoreAdapter(unittest.TestCase):
    def test_tmp(self):
        b=pa.Tensor.from_numpy(np.zeros((3,4)))
        np_arr = np.zeros(100).reshape(5,-1)
        data = [
            # pa.array([1, 2, 3, 4]),
            # pa.array(['foo', 'bar', 'baz', None]),
            np_arr.reshape(-1)
            # pa.array([True, None, False, True])
        ]
        # batch = pa.RecordBatch.from_arrays(data, ['f0', 'f1', 'f2'])
        batch = pa.RecordBatch.from_arrays(data, ['f0'],metadata={"s":"3:4"})

        print(batch.schema)
        path = "./test_tmp1"
        with pa.ipc.new_file(path, batch.schema) as w:
            w.write_batch(batch)

        with pa.ipc.open_file(path) as fd:
            # df =fd.read_pandas()
            df =fd.get_batch(0)
            df[0].to_numpy().reshape(5,-1)
            print(df.schema)

        print(pa.total_allocated_bytes())

    def test_frame_batch(self):
        fr = FrameBatch.from_pandas(pd.DataFrame.from_dict({"a":[1,2]}))
        print(fr._data)
        t1 = TensorBatch.from_numpy(np.zeros([3,4]))
        print(TensorBatch.from_frame(t1.to_frame())._data)

    def test_pandas_dataframe_rw(self):
        df1 = pd.DataFrame.from_dict({"f_int": [1, 2], "f_double": [None, 2.0], "f_str": ["str1", None], "f_none": [None, None]})
        df2 = pd.DataFrame.from_dict({"f_int": [None, 2, None], "f_double": [1.0, 2.0, 3.0], "f_str": ["str1", None, "str3"], "f_none": [None, None, None]})
        df3 = pd.DataFrame.from_dict({"f_int": [1, 2, 3], "f_double": [1.0, 2.0, 3.0], "f_str": ["str1", None, "str3"], "f_none": [None, None, None]})
        options = {}
        options['path'] = "/tmp/test_py_rf/pandas_df"
        with create_frame_adapter(options) as adapter:
            adapter.write_all([df1, df3])

        with create_frame_adapter(options) as adapter:
            fbs = adapter.read_all()
            for fb in fbs:
                data = fb.to_pandas()
                print(data)

    def test_read(self):
        options = {}
        options['path'] = '/Users/max-webank/git/eggroll-py-rf/data/ROLL_FRAME_FILE/test_rf_ns/test_rf_name_20200619.094320.566/0'
        with create_frame_adapter(options) as adapter:
            fbs = adapter.read_all()
            for fb in fbs:
                data = fb.to_pandas()
                print(data)

