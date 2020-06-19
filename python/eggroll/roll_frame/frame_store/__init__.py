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

from typing import Iterable

from eggroll.core.constants import StoreTypes
from eggroll.core.meta_model import ErPartition
# TODO:0: move to a common path
from eggroll.roll_pair.utils.pair_utils import get_db_path


class FrameAdapter(object):
    def read_all(self):
        raise NotImplementedError()

    def write_all(self, batches: Iterable, options: dict = None):
        raise NotImplementedError()


# TODO:0: eliminate duplicate with pair's create_adapter
def create_adapter(er_partition: ErPartition, options: dict = None):
    if options is None:
        options = {}
    options['store_type'] = er_partition._store_locator._store_type
    options['path'] = get_db_path(er_partition)
    options['er_partition'] = er_partition
    return create_frame_adapter(options=options)


def create_frame_adapter(options: dict) -> FrameAdapter:
    if options is None:
        options = {}
    store_type = options.get('store_type', StoreTypes.ROLLFRAME_FILE)

    ret = None
    if store_type == StoreTypes.ROLLFRAME_FILE:
        from eggroll.roll_frame.frame_store.file_adapter import FileFrameAdapter
        ret = FileFrameAdapter(options=options)
    else:
        raise NotImplementedError(options)

    return ret
