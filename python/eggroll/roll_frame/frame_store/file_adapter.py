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

import os
import pyarrow as pa

from typing import Iterable

from eggroll.roll_frame.frame_store import FrameAdapter
from eggroll.roll_frame import FrameBatch


class FileFrameAdapter(FrameAdapter):
    RF_FILE_NAME = 'data.erf'

    def __init__(self, options):
        self.path = options["path"]
        os.makedirs(self.path, exist_ok=True)
        self.path = f'{self.path}/{FileFrameAdapter.RF_FILE_NAME}'

    def read_all(self, options: dict = None):
        if options is None:
            options = {}

        with pa.ipc.open_file(self.path) as reader:
            num_record_batches = reader.num_record_batches
            schema = reader.schema
            for i in range(num_record_batches):
                yield FrameBatch(reader.get_record_batch(i), schema)

    def write_all(self, batches: Iterable, options: dict = None):
        if options is None:
            options = {}
        batch_iter = iter(batches)
        try:
            first = FrameBatch(next(batch_iter))
            schema = first._schema
        except StopIteration as e:
            return

        with pa.ipc.new_file(self.path, schema=schema) as writer:
            writer.write_batch(first._data)
            for batch in batch_iter:
                writer.write_batch(FrameBatch(batch)._data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass