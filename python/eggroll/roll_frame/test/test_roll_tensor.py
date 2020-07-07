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

        rt = self.ctx.load(namespace=self.namespace, name=self.name_1p, options=self.options_3p)
        result = rt.with_stores(_max)

        self.assertTrue((np.mat("9, 10, 11, 12") == result[0][1]).all())
        print(result)

    def test_put_all_3p(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        rt.put_all(self.mat3)

    def test_get_all_3p(self):
        rt = self.ctx.load(namespace=self.namespace, name=self.name_3p, options=self.options_3p)
        tensor = rt.get_all()

        self.assertTrue((self.mat3 == tensor.to_numpy()).all())
        print(tensor.to_numpy())