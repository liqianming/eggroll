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
#
#

import numpy as np
import pandas as pd
import pyarrow as pa
import unittest
import threading
import time

from concurrent.futures.thread import ThreadPoolExecutor

from eggroll.core.constants import StoreTypes
from eggroll.core.utils import time_now
from eggroll.core.session import ErSession
from eggroll.roll_frame.test.roll_tensor_test_assets import get_debug_test_context
from eggroll.roll_frame.frame_store import create_frame_adapter, create_adapter
from eggroll.utils.log_utils import get_logger
from eggroll.roll_frame import FrameBatch, TensorBatch

L = get_logger()


class TestRollTensorBase(unittest.TestCase):
    mat1 = np.mat("1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12")
    mat3 = np.mat("1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16")
    vec = np.mat("1, 2, 3, 4")
    name_1p = f'test_rt_name_1p'
    name_3p = f'test_rt_name_3p'
    namespace = 'test_rt_namespace'
    options_1p = {'total_partitions': 1}
    options_3p = {'total_partitions': 3}

    def setUp(self):
        self.ctx = get_debug_test_context()

    def tearDown(self) -> None:
        print("stop test session")
        # self.ctx.get_session().stop()

    def test_put_all_1p(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_1p, options=self.options_1p)
        rt.put_all(self.mat1)

    def test_get_all_1p(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_1p, options=self.options_1p)
        tensor = rt.get_all()

        self.assertTrue((self.mat1 == tensor.to_numpy()).all())
        print(tensor.to_numpy())

    def test_max_1p(self):
        def _max(task):
            with create_adapter(task._inputs[0]) as input_adapter:
                for batch in input_adapter.read_all():
                    result = TensorBatch(batch).to_numpy().max(axis=0)
                    return result

        rt = self.ctx.load(namespace=self.namespace, name=self.name_1p, options=self.options_1p)
        result = rt.with_stores(_max)

        self.assertTrue((np.max(self.mat1, axis=0) == result[0][1]).all())
        print(result)

    def test_put_all_3p(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        rt.put_all(self.mat3)

    def test_get_all_3p(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        tensor = rt.get_all()

        self.assertTrue((self.mat3 == tensor.to_numpy()).all())
        print(tensor.to_numpy())

    def test_max_3p(self):
        def _merge(partial_result):
            result = None
            shape = None
            for r in partial_result:
                if result is not None:
                    result = np.max(np.concatenate([result, r]), axis=0).reshape(shape)
                else:
                    result = r
                    shape = r.shape

            return TensorBatch(result)

        def _max(task):
            with create_adapter(task._inputs[0]) as input_adapter:
                for batch in input_adapter.read_all():
                    result = TensorBatch(batch).to_numpy().max(axis=0)
                    return result.reshape(1, result.shape[0])

        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        result = rt.with_stores(func=_max, merge_func=_merge)

        self.assertTrue((np.max(self.mat3, axis=0) == result.to_numpy()).all())
        print(result)

    def test_min_3p(self):
        def _merge(partial_result):
            result = None
            shape = None
            for r in partial_result:
                if result is not None:
                    result = np.min(np.concatenate([result, r]), axis=0).reshape(shape)
                else:
                    result = r
                    shape = r.shape

            return TensorBatch(result)

        def _min(task):
            with create_adapter(task._inputs[0]) as input_adapter:
                for batch in input_adapter.read_all():
                    result = TensorBatch(batch).to_numpy().min(axis=0)
                    return result.reshape(1, result.shape[0])

        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        result = rt.with_stores(func=_min, merge_func=_merge)

        self.assertTrue((np.min(self.mat3, axis=0) == result.to_numpy()).all())
        print(result.to_numpy())

    def test_avg(self):
        def _merge(partial_result):
            result = None
            partial_shape = None
            total_rows = 0

            for r in partial_result:
                if result is not None:
                    result = np.sum(np.concatenate([result, r[0]]), axis=0).reshape(partial_shape)
                else:
                    result = r[0]
                    partial_shape = r[0].shape
                total_rows += r[1]

            result = np.divide(result, total_rows)
            return TensorBatch(result)

        def _sum_rows(task):
            with create_adapter(task._inputs[0]) as input_adapter:
                for batch in input_adapter.read_all():
                    tb = TensorBatch(batch)
                    sum = tb.to_numpy().sum(axis=0)
                    return (sum.reshape(1, sum.shape[0]), tb._shape[0])

        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        result = rt.with_stores(func=_sum_rows, merge_func=_merge)

        self.assertTrue((np.average(self.mat3, axis=0) == result.to_numpy()).all())
        print(result.to_numpy())

    def test_std(self):
        def _merge(partial_result):
            sum_of_square_ = None
            square_of_sum_ = None
            partial_shape = None
            total_rows = 0

            for r in partial_result:
                if sum_of_square_ is not None:
                    sum_of_square_ = np.sum([sum_of_square_, r[0]], axis=0).reshape(partial_shape)
                    square_of_sum_ = np.sum([square_of_sum_, r[1]], axis=0).reshape(partial_shape)
                else:
                    sum_of_square_ = r[0]
                    square_of_sum_ = r[1]
                    partial_shape = r[0].shape
                total_rows += r[2]

            sum_of_square_ = sum_of_square_ / total_rows
            square_of_sum_ = (square_of_sum_ * square_of_sum_) / (total_rows * total_rows)
            result = np.sqrt(sum_of_square_ - square_of_sum_)
            return TensorBatch(result)

        def _std(task):
            with create_adapter(task._inputs[0]) as input_adapter:
                for batch in input_adapter.read_all():
                    tb = TensorBatch(batch)
                    npb = tb.to_numpy()
                    sum_of_square = np.sum(np.power(npb, 2), axis=0).reshape(1, tb._shape[1])
                    sum_ = np.sum(npb, axis=0).reshape(1, tb._shape[1])
                    return (sum_of_square, sum_, tb._shape[0])

        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        result = rt.with_stores(func=_std, merge_func=_merge)

        self.assertTrue((np.std(self.mat3, axis=0) == result.to_numpy()).all())
        print(result.to_numpy())

    def test_local_max(self):
        result = TensorBatch(self.mat3).to_numpy().max(axis=0)
        self.assertTrue((np.max(self.mat3, axis=0) == result).all())
        print(result)

    def test_sub_dist_local(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        tb = TensorBatch(self.vec)

        result = rt - tb
        self.assertTrue((result.get_all().to_numpy() == self.mat3 - self.vec).all())
        print(result.get_all().to_numpy())

    def test_truediv_dist_local(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        tb = TensorBatch(self.vec)
        result = rt / tb
        self.assertTrue((result.get_all().to_numpy() == self.mat3 / self.vec).all())
        print(result.get_all().to_numpy())