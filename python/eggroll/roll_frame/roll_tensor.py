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
import numpy as np
import pyarrow as pa

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
from eggroll.roll_frame.roll_frame import RollFrameContext, RollFrame
from eggroll.utils.log_utils import get_logger

L = get_logger()


class RollTensorContext(object):
    def __init__(self, session: ErSession, rf_ctx : RollFrameContext = None):
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


class RollTensor(object):
    RUN_TASK_URI = RollFrame.RUN_TASK_URI
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
    def put_all(self, tensor):
        frame = TensorBatch.from_numpy(tensor).to_frame()
        rf = self._rf.put_all([frame])
        return RollTensor(rf.get_store(), self.ctx)

    @_method_profile_logger
    def get_all(self):
        fb = self._rf.get_all()
        return TensorBatch.from_frame(frame=fb)

    @_method_profile_logger
    def with_stores(self, func, merge_func=None, others=None, options: dict = None):
        return self._rf.with_stores(func=func,
                                  merge_func=merge_func,
                                  others=others,
                                  options=options)

