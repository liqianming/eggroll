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

import pandas as pd
import pyarrow as pa
import unittest
import threading
import time

from concurrent.futures.thread import ThreadPoolExecutor

from eggroll.core.constants import StoreTypes
from eggroll.core.utils import time_now
from eggroll.roll_frame.test.roll_frame_test_assets import get_debug_test_context
from eggroll.roll_frame.frame_store import create_frame_adapter, create_adapter
from eggroll.utils.log_utils import get_logger
from eggroll.roll_frame import FrameBatch

L = get_logger()


class TestRollFrameBase(unittest.TestCase):
    df2 = pd.DataFrame.from_dict({"f_int": [1, 2, 3], "f_double": [1.0, 2.0, 3.0]})
    df3 = pd.DataFrame.from_dict({"f_int": [1, 2, 3], "f_double": [1.0, 2.0, 3.0], "f_str": ["str1", None, "str3"], "f_none": [None, None, None]})
    df4 = pd.DataFrame.from_dict({"f_int": [-3, 0, 6, 4], "f_double": [-1.0, 2.0, 3.0, 2.4], "f_str": ["str4", None, "str3", "str6"], "f_none": [None, None, None, None]})

    def setUp(self):
        self.ctx = get_debug_test_context()
        self.namespace = 'test_rf_ns'
        self.name_2p_numeric = "test_rf_name_2p_numeric"
        self.name_1p = 'test_rf_name_1p'
        self.name_2p = 'test_rf_name_2p'

    def tearDown(self) -> None:
        print("stop test session")
        # self.ctx.get_session().stop()

    def test_put_all(self):
        rf = self.ctx.load(name=self.name_2p_numeric, namespace=self.namespace, options={"total_partitions": 2})
        rf.put_all(self.df2)

        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace, options={"total_partitions": 2})
        rf.put_all(self.df3)

        rf = self.ctx.load(name=self.name_1p, namespace=self.namespace, options={"total_partitions": 1})
        rf.put_all(self.df3)

    def test_get_all(self):
        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace)
        local_rf = rf.get_all()
        self.assertTrue(self.df3.equals(local_rf.to_pandas()))
        print(local_rf.to_pandas())

    def test_max(self):
        def agg(iterable_frames):
            frame = FrameBatch.concat(iterable_frames)
            return frame.to_pandas().max()

        def ef_max(task):
            # def batch_max_generator(task):
            #     with create_adapter(task._inputs[0]) as input_adapter:
            #         for batch in input_adapter.read_all():
            #             yield batch.to_pandas().max().to_frame().transpose()
            # result = agg(batch_max_generator(task))

            with create_adapter(task._inputs[0]) as input_adapter:
                result = agg(batch.to_pandas().max().to_frame().transpose() for batch in input_adapter.read_all())

                return result

        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace)

        rf_results = rf.with_stores(ef_max)
        result = agg(r[1].to_frame().transpose() for r in rf_results)
        self.assertEqual(result['f_double'], 3.0)
        return result

    def test_max_with_merge(self):
        def merger(iterable_series):
            result = None
            for series in iterable_series:
                if result is not None:
                    result = pd.concat([result.to_frame().transpose(),
                                        series.to_frame().transpose()]).max()
                else:
                    result = series
            return result

        def ef_max(task):
            with create_adapter(task._inputs[0]) as input_adapter:
                result = merger(batch.to_pandas().max() for batch in input_adapter.read_all())

                return result

        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace)

        result = rf.with_stores(ef_max, merger)
        self.assertEqual(result['f_double'], 3.0)
        return result

    def test_max_with_agg_2p(self):
        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace)

        result = rf.agg('max')

        print(result.to_pandas())

    def test_max_with_agg_2p_list(self):
        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace)

        result = rf.agg(['max'])

        print(result.to_pandas())

    def test_max_with_agg_1p(self):
        rf = self.ctx.load(name=self.name_1p, namespace=self.namespace)

    def test_std_with_agg(self):
        rf = self.ctx.load(name=self.name_2p_numeric, namespace=self.namespace)

        result = rf.agg('std')
        print(result.to_pandas())

    def test_with_store(self):
        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace)

        def get_max_threads_count(task):
            def foo():
                time.sleep(5)

            ret = None
            for i in range(1000):
                try:
                    t = threading.Thread(target=foo)
                    t.start()
                except:
                    ret = threading.active_count()
                    L.exception("============= exception caught. active threads", ret)

            ret = threading.active_count()
            L.info("============= exit normally. active threads", ret)

            return ret

        result = rf.with_stores(get_max_threads_count)
        print(result)
