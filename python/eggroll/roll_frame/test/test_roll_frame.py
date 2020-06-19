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

from concurrent.futures.thread import ThreadPoolExecutor

from eggroll.core.constants import StoreTypes
from eggroll.core.utils import time_now
from eggroll.roll_frame.test.roll_frame_test_assets import get_debug_test_context


class TestRollFrameBase(unittest.TestCase):
    def setUp(self):
        self.ctx = get_debug_test_context()

    def tearDown(self) -> None:
        print("stop test session")
        # self.ctx.get_session().stop()

    def test_put_all(self):
        df3 = pd.DataFrame.from_dict({"f_int": [1, 2, 3], "f_double": [1.0, 2.0, 3.0], "f_str": ["str1", None, "str3"], "f_none": [None, None, None]})
        rf = self.ctx.load('test_rf_ns', f'test_rf_name_1', options={"total_partitions": 2})
        rf.put_all(df3)

    def test_get_all(self):
        rf = self.ctx.load('test_rf_ns', f'test_rf_name_1', options={"total_partitions": 2})
        local_rf = rf.get_all()
        print(local_rf.to_pandas())