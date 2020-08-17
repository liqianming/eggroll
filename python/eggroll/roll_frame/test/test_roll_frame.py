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
import hashlib
import operator

import threading
import time
import unittest

import pandas as pd

from eggroll.roll_frame import FrameBatch
from eggroll.roll_frame.frame_store import create_adapter
from eggroll.roll_frame.test.roll_frame_test_assets import \
    get_debug_test_context
from eggroll.utils.log_utils import get_logger

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

    def test_parallelize(self):
        rf = self.ctx.parallelize(self.df2, options={'total_partitions': 3})
        print(rf.get_name(), rf.get_namespace())
        print(rf.get_all().to_pandas())

    def test_put_all(self):
        rf = self.ctx.load(name=self.name_2p_numeric, namespace=self.namespace, options={"total_partitions": 2})
        rf.put_all(self.df2)

        rf = self.ctx.load(name=self.name_2p, namespace=self.namespace, options={"total_partitions": 2})
        rf.put_all(self.df3)

        rf = self.ctx.load(name=self.name_1p, namespace=self.namespace, options={"total_partitions": 1})
        rf.put_all(self.df3)

    @staticmethod
    def __hash(x):
        return int(hashlib.sha256(bytes(str(x), encoding='utf-8')).hexdigest()[:8], base=16)

    def test_hash(self):
        def hash_frame(task):
            with create_adapter(task._inputs[0]) as input_adapter, create_adapter(task._outputs[0]) as output_adapter:
                for batch in input_adapter.read_all():
                    print(f"batch={batch.to_pandas()}")
                    if not batch.to_pandas().empty:
                        print(f"start write data={batch.to_pandas().applymap(TestRollFrameBase.__hash)}")
                        output_adapter.write_all([batch.to_pandas().applymap(TestRollFrameBase.__hash)])
                    else:
                        output_adapter.write_all(pd.DataFrame(data=None))

        rf = self.ctx.load(name=self.name_2p_numeric+"_hash", namespace=self.namespace, options={"total_partitions": 1})
        rf.put_all(self.df2)
        rf.with_stores(hash_frame)

    def test_mul(self):
        df1 = pd.DataFrame.from_dict({"f_int": [1, 2, 3], "f_double": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame.from_dict({"f_int": [1, 2, 3], "f_double": [1.0, 2.0, 3.0]})
        rf1 = self.ctx.load(name="mul_self", namespace=self.namespace, options={"total_partitions": 1})
        rf1.put_all(df1)
        rf2 = self.ctx.load(name="mul_other", namespace=self.namespace, options={"total_partitions": 1})
        rf2.put_all(df2)
        rf2_data = rf2.get_all().to_pandas()
        import operator
        def multiply_wrapper(task):
            with create_adapter(task._inputs[0]) as input_adapter, create_adapter(task._outputs[0]) as output_adapter:
                for batch in input_adapter.read_all():
                    print(f"batch={batch.to_pandas()}")
                    if not batch.to_pandas().empty:
                        mul_res = operator.mul(batch.to_pandas(), rf2_data)
                        output_adapter.write_all([mul_res])
                    else:
                        output_adapter.write_all(pd.DataFrame(data=None))
                return task._outputs[0]
        res = rf1.with_stores(multiply_wrapper)
        print(self.ctx.load(namespace=res[0][1]._store_locator._namespace, name=res[0][1]._store_locator._name).get_all().to_pandas())

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

    def test_agg(self):
        rf = self.ctx.load(name=self.name_2p_numeric, namespace=self.namespace)

        print(rf.get_all().to_pandas())
        result = rf.agg(['std', 'max', 'min', 'count'])
        print(result.to_pandas())
        print('-------------')

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
