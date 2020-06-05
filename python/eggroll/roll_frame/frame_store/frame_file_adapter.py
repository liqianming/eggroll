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

from eggroll.roll_frame import FrameBatch



class FrameFileAdapter:
    def __init__(self, options):
        self.path = options["path"]

    def read_all(self):
        with pa.ipc.open_file(self.path) as fd:
            for batch in fd:
                yield FrameBatch(batch)

    def write_all(self, batches):
        batches = iter(batches)
        try:
            first = next(batches)
        except StopIteration:
            return

        with pa.ipc.new_file(self.path, first.schema) as w:
            w.write_batch(first._data)
            for batch in batches:
                w.write_batch(batch)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass