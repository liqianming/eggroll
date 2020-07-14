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

import inspect
import pandas as pd
import numpy as np
import pyarrow as pa
import uuid
from typing import Union

from eggroll.core.aspects import _method_profile_logger
from eggroll.core.client import CommandClient
from eggroll.core.command.command_model import CommandURI
from eggroll.core.conf_keys import SessionConfKeys
from eggroll.core.constants import SerdesTypes, StoreTypes
from eggroll.roll_pair import create_serdes
from eggroll.core.meta_model import ErStoreLocator, ErJob, ErStore, ErFunctor, \
    ErTask, ErPair, ErPartition
from eggroll.core.session import ErSession
from eggroll.roll_frame import TensorBatch
from eggroll.roll_frame.frame_store import create_adapter
from eggroll.roll_frame.roll_frame import RollFrameContext, RollFrame
from eggroll.utils.log_utils import get_logger
from eggroll.core.serdes.eggroll_serdes import DefaultArrowSerdes

L = get_logger()


class RollTensorContext(object):
    def __init__(self, session: ErSession, rf_ctx: RollFrameContext = None):
        if session is None and rf_ctx is None:
            raise ValueError(f'both session and context are None')
        if rf_ctx:
            self.rf_ctx = rf_ctx
            self.__session = rf_ctx.get_session()
        else:
            self.rf_ctx = RollFrameContext(session)
            self.__session = session

        self.session_id = session.get_session_id()
        self.default_store_type = StoreTypes.ROLLFRAME_FILE

        self.deploy_mode = session.get_option(SessionConfKeys.CONFKEY_SESSION_DEPLOY_MODE)
        self.__session_meta = session.get_session_meta()
        self.__command_client = CommandClient()

    def set_store_type(self, store_type: str):
        self.default_store_type = store_type

    def get_session(self):
        return self.__session

    def route_to_egg(self, partition: ErPartition):
        return self.__session.route_to_egg(partition)

    def populate_processor(self, store: ErStore):
        return self.__session.populate_processor(store)

    def load(self, namespace=None, name=None, options: dict = None):
        rf = self.rf_ctx.load(namespace=namespace, name=name, options=options)

        return RollTensor(rf.get_store(), self)

    def parallelize(self, data, options: dict = None):
        if options is None:
            options = {}
        namespace = options.get("namespace", self.session_id)
        name = options.get("name", f'rt_{uuid.uuid1()}')
        options['store_type'] = options.get("store_type", StoreTypes.ROLLFRAME_FILE)
        create_if_missing = options.get("create_if_missing", True)

        rt = self.load(namespace=namespace, name=name, options=options)
        return rt.put_all(data, options=options)




class RollTensor(object):
    RUN_TASK_URI = RollFrame.RUN_TASK_URI


    def dispatch(self, other):
        op = inspect.getxxx # add
        new_m = op + "loc" + "roll"
        return getattr(self, new_m)(other)

    def __add__(self, other):
        self.dispatch(other)
        # loc & shape=[1] => bcast

        # loc & shape = [x,y] => scatter => elem wise

    def __add__loc_roll(self, other):
        bcast

    def __add__roll_roll(self, other):
        scatter

    def __init__(self, er_store: ErStore, rt_ctx: RollTensorContext):
        if not rt_ctx:
            raise ValueError('rt_ctx cannot be None')
        self.__store = er_store
        self.ctx = rt_ctx
        self.__command_client = CommandClient()
        self.functor_serdes = create_serdes(SerdesTypes.CLOUD_PICKLE)
        self.__session_id = self.ctx.session_id
        self.destroyed = False
        self._rf = RollFrame(er_store, rt_ctx)

    def get_partitions(self):
        return self.__store._store_locator._total_partitions

    def get_name(self):
        return self.__store._store_locator._name

    def get_namespace(self):
        return self.__store._store_locator._namespace

    def get_store(self):
        return self.__store

    def get_store_type(self):
        return self.__store._store_locator._store_type

    def _submit_job(self,
            job: ErJob,
            command_uri: CommandURI = RUN_TASK_URI,
            create_output_if_missing: bool = True):
        futures = self.ctx.get_session().submit_job(
                job=job,
                output_types=[ErPair],
                command_uri=command_uri,
                create_output_if_missing=create_output_if_missing)

        return futures

    def _wait_job_finished(self, futures: list):
        results = list()
        for future in futures:
            results.append(future.result())

        return results

    def _run_job(self,
            job: ErJob,
            command_uri: CommandURI = RollFrame.RUN_TASK_URI,
            create_output_if_missing: bool = True):
        futures = self._submit_job(
                job=job,
                command_uri=command_uri,
                create_output_if_missing=create_output_if_missing)

        return self._wait_job_finished(futures)

    @_method_profile_logger
    def put_all(self, tensor, options: dict = None):
        if options is None:
            options = {}
        def gen_axis_0_split(tensor_batch: TensorBatch, total_partitions):
            npt = tensor_batch.to_numpy()
            axis_0_per_block = npt.shape[0] // total_partitions
            axis_0_extra = npt.shape[0] % total_partitions

            axis_0_start = 0
            for i in range(total_partitions):
                axis_0_end = min(axis_0_start + axis_0_per_block + (1 if i < axis_0_extra else 0), npt.shape[0])
                batch = npt[axis_0_start:axis_0_end]
                fb = TensorBatch(data=batch, block_start=(axis_0_start, 0), block_end=(axis_0_end - 1, npt.shape[1] - 1)).to_frame()
                axis_0_start = axis_0_end
                yield fb

        tensor_batch = TensorBatch(tensor)
        rf = self._rf.put_all(gen_axis_0_split(
                tensor_batch=tensor_batch,
                total_partitions=self.get_partitions()))
        return RollTensor(rf.get_store(), self.ctx)

    @_method_profile_logger
    def get_all(self, options: dict = None):
        if options is None:
            options = {}
        fbs = self._rf.get_all()
        start_coordinate = TensorBatch._bytes_to_int_tuple(fbs[0]._schema.metadata[TensorBatch.META_BLOCK_START_KEY])
        end_coordinate = TensorBatch._bytes_to_int_tuple(fbs[-1]._schema.metadata[TensorBatch.META_BLOCK_END_KEY])
        full_shape = tuple(np.add(np.subtract(end_coordinate, start_coordinate), (1, 1)))

        arrays = []
        for f in fbs:
            a = TensorBatch(f).to_numpy()
            arrays.append(a)

        result = np.concatenate(arrays).reshape(full_shape)

        return TensorBatch(result)

    @_method_profile_logger
    def with_stores(self, func, merge_func=None, others=None, options: dict = None):
        return self._rf.with_stores(func=func,
                                    merge_func=merge_func,
                                    others=others,
                                    options=options)

    def __bin_op(self, other):
        real_func_frame = inspect.currentframe().f_back
        target_func_name = '_'.join([real_func_frame.f_code.co_name,
                                     tensor_short_names[type(self).__qualname__],
                                     tensor_short_names[type(other).__qualname__]])
        return getattr(self, f'_{self.__class__.__name__}{target_func_name}')(other)

    def __sub__(self, other):
        return self.__bin_op(other)

    def __truediv__(self, other):
        return self.__bin_op(other)

    def __sub___dist_local(self, other: TensorBatch):
        def _sub(task):
            with create_adapter(task._inputs[0]) as input_adapter, \
                    create_adapter(task._outputs[0]) as output_adapter:
                for batch in input_adapter.read_all():
                    tb = TensorBatch(batch)
                    np_result = tb.to_numpy() - other.to_numpy()
                    result = TensorBatch(data=np_result, block_start=tb._block_start, block_end=tb._block_end)
                    output_adapter.write_all([result.to_frame()])

            return task
        result = self._rf.with_stores(func=_sub)

        return RollTensor(er_store=result[0][1]._job._outputs[0], rt_ctx=self.ctx)

    def __truediv___dist_local(self, other: TensorBatch):
        def _truediv(task):
            with create_adapter(task._inputs[0]) as input_adapter, \
                    create_adapter(task._outputs[0]) as output_adapter:
                for batch in input_adapter.read_all():
                    tb = TensorBatch(batch)
                    np_result = tb.to_numpy() / other.to_numpy()
                    result = TensorBatch(data=np_result, block_start=tb._block_start, block_end=tb._block_end)
                    output_adapter.write_all([result.to_frame()])
            return task
        result = self._rf.with_stores(func=_truediv)

        return RollTensor(er_store=result[0][1]._job._outputs[0], rt_ctx=self.ctx)


tensor_short_names = {RollTensor.__qualname__: "dist",
                      TensorBatch.__qualname__: "local"}
