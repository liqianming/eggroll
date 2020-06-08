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
import operator
import os
import uuid
from concurrent.futures import wait, FIRST_EXCEPTION
from queue import Queue
from threading import Thread

from eggroll.core.aspects import _method_profile_logger
from eggroll.core.client import CommandClient
from eggroll.core.command.command_model import CommandURI
from eggroll.core.conf_keys import SessionConfKeys, RollPairConfKeys
from eggroll.core.constants import StoreTypes, SerdesTypes, PartitionerTypes, \
    SessionStatus
from eggroll.core.datastructure.broker import FifoBroker
from eggroll.core.io.io_utils import get_db_path
from eggroll.core.meta_model import ErStoreLocator, ErJob, ErStore, ErFunctor, \
    ErTask, ErPair, ErPartition
from eggroll.core.serdes import cloudpickle
from eggroll.core.session import ErSession
from eggroll.core.utils import generate_job_id, generate_task_id
from eggroll.core.utils import string_to_bytes, hash_code
from eggroll.roll_frame.frame_store import FrameStore
from eggroll.roll_pair import create_serdes
from eggroll.roll_pair.transfer_pair import TransferPair, BatchBroker
from eggroll.roll_pair.utils.gc_utils import GcRecorder
from eggroll.roll_pair.utils.pair_utils import partitioner
from eggroll.utils.log_utils import get_logger
import pandas as pd
import numpy as np
import pyarrow as pa
L = get_logger()


class RollFrameContext(object):

    def __init__(self, session: ErSession):
        if session.get_session_meta()._status != SessionStatus.ACTIVE:
            raise Exception(f"session:{session.get_session_id()} is not ACTIVE. current status={session.get_session_meta()._status}")
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
        if options is None:
            options = {}
        store_type = options.get('store_type', self.default_store_type)
        total_partitions = options.get('total_partitions', None)
        no_partitions_param = False
        if total_partitions is None:
            no_partitions_param = True
            total_partitions = 1

        partitioner = options.get('partitioner', PartitionerTypes.BYTESTRING_HASH)
        create_if_missing = options.get('create_if_missing', True)
        # todo:1: add combine options to pass it through
        store_options = self.__session.get_all_options()
        store_options.update(options)
        final_options = store_options.copy()

        # TODO:1: tostring in er model
        if 'create_if_missing' in final_options:
            del final_options['create_if_missing']
        if 'total_partitions' in final_options:
            del final_options['total_partitions']
        if 'name' in final_options:
            del final_options['name']
        if 'namespace' in final_options:
            del final_options['namespace']
        # TODO:0: add 'error_if_exist, persistent / default store type'
        L.info("final_options:{}".format(final_options))
        store = ErStore(
            store_locator=ErStoreLocator(
                store_type=store_type,
                namespace=namespace,
                name=name,
                total_partitions=total_partitions,
                partitioner=partitioner),
            options=final_options)

        if create_if_missing:
            result = self.__session._cluster_manager_client.get_or_create_store(store)
        else:
            result = self.__session._cluster_manager_client.get_store(store)
            if result is None:
                raise EnvironmentError(
                    "result is None, please check whether the store:{} has been created before".format(store))

        if not no_partitions_param and result._store_locator._total_partitions != 0 \
                and total_partitions != result._store_locator._total_partitions:
            raise ValueError(f"store:{result._store_locator._name} input total_partitions:{total_partitions}, "
                             f"output total_partitions:{result._store_locator._total_partitions}, must be the same")

        return RollFrame(self.populate_processor(result), self)

    # TODO:1: separates load parameters and put all parameters
    def parallelize(self, data, options: dict = None):
        if options is None:
            options = {}
        namespace = options.get("namespace", None)
        name = options.get("name", None)
        options['store_type'] = options.get("store_type", StoreTypes.ROLLPAIR_IN_MEMORY)
        create_if_missing = options.get("create_if_missing", True)

        if namespace is None:
            namespace = self.session_id
        if name is None:
            name = str(uuid.uuid1())
        rp = self.load(namespace=namespace, name=name, options=options)
        return rp.parallelize(data, options=options)

    '''store name only supports full name and reg: *, *abc ,abc* and a*c'''
    def cleanup(self, namespace, name, options: dict = None):
        if not namespace:
            raise ValueError('namespace cannot be blank')
        L.info(f'cleaning up namespace={namespace}, name={name}')
        if options is None:
            options = {}
        total_partitions = options.get('total_partitions', 1)
        partitioner = options.get('partitioner', PartitionerTypes.BYTESTRING_HASH)
        store_serdes = options.get('serdes', self.default_store_serdes)

        if name == '*':
            pass
        else:
            pass


def _create_store_adapter(er_partition, options: dict = None):
    if options is None:
            options = {}
    store_locator = er_partition._store_locator
    options['store_type'] = er_partition._store_locator._store_type
    options['path'] = "/".join([os.environ["EGGROLL_HOME"], store_locator._store_type,
                                store_locator._namespace, store_locator._name, str(er_partition._id)])
    options['er_partition'] = er_partition
    return FrameStore.init(options)


class RollFrame(object):
    EGG_FRAME_URI_PREFIX = 'v1/egg-pair'
    RUN_TASK = 'runTask'
    RUN_TASK_URI = CommandURI(f'{EGG_FRAME_URI_PREFIX}/{RUN_TASK}')
    SERIALIZED_NONE = cloudpickle.dumps(None)

    def __setstate__(self, state):
        pass

    def __getstate__(self):
        pass

    def __init__(self, er_store: ErStore, rp_ctx):
        if not rp_ctx:
            raise ValueError('rp_ctx cannot be None')
        self.__store = er_store
        self.ctx = rp_ctx
        self.__command_client = CommandClient()
        self.functor_serdes =create_serdes(SerdesTypes.CLOUD_PICKLE)
        self.partitioner = partitioner(hash_code, self.__store._store_locator._total_partitions)
        self.__session_id = self.ctx.session_id
        self.destroyed = False

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

    def _run_job(self,
                 job: ErJob,
                 output_types: list = None,
                 command_uri: CommandURI = RUN_TASK_URI,
                 create_output_if_missing: bool = True):
        futures = self.ctx.get_session().submit_job(
            job=job,
            output_types=output_types,
            command_uri=command_uri,
            create_output_if_missing=create_output_if_missing)

        results = list()
        for future in futures:
            results.append(future.result())

        return results

    def put_all(self, frame):
        def _inner_put_all(parts):
            part = parts[0]
            with _create_store_adapter(part) as input_store:
                # TODO: merge batch
                input_store.write_all(TransferFrame.get(tag))
        if isinstance(frame, pd.DataFrame):
            for batch in frame.split(by_column_value=True):
                route_to_egg = xx
                get_egg(route_to_egg).send(batch)
        else:
            raise NotImplementedError("doing")
        return self.with_stores(_inner_put_all)

    def shuffle(self):
        def _inner_shuffle(parts):
            in_part, out_part = parts[0], parts[1]
            with _create_store_adapter(in_part) as in_store, _create_store_adapter(out_part) as out_store:
                # TODO: merge batch
                input_store.write_all(TransferFrame.get(tag))
                for frame in in_store.read_all():
                    for batch in frame.split(by_column_value=True):
                        route_to_egg = xx
                        get_egg(route_to_egg).send(batch)
        return self.with_stores(_inner_shuffle)

    @_method_profile_logger
    def count(self):
        def _inner_count(parts):
            part = parts[0]

            with _create_store_adapter(part) as input_store:
                count = 0
                for batch in input_store.read_all():
                    count += batch.to_pandas().count()[0]
            L.debug(f"close_store_adatper:{count}")

        def _count_merge(results):
            ret = 0
            for pair in results:
                ret += pair[1]
            return ret

        return self.with_stores(_inner_count, merge_func=_count_merge)

    # todo:1: move to command channel to utilize batch command
    @_method_profile_logger
    def destroy(self):
        if len(self.ctx.get_session()._cluster_manager_client.get_store(self.get_store())._partitions) == 0:
            L.info(f"store:{self.get_store()} has been destroyed before")
            raise ValueError(f"store:{self.get_store()} has been destroyed before")
        total_partitions = self.__store._store_locator._total_partitions

        job = ErJob(id=generate_job_id(self.__session_id, "destroy"),
                    name="destroy",
                    inputs=[self.__store],
                    outputs=[self.__store],
                    functors=[])

        task_results = self._run_job(job=job, create_output_if_missing=False)
        self.ctx.get_session()._cluster_manager_client.delete_store(self.__store)
        self.destroyed = True

    @_method_profile_logger
    def with_stores(self, func, others=None, options: dict=None, merge_func=None):
        if options is None:
            options = {}
        tag = "withStores"
        if others is None:
            others = []
        total_partitions = self.get_partitions()
        for other in others:
            if other.get_partitions() != total_partitions:
                raise ValueError(f"diff partitions: expected:{total_partitions}, actual:{other.get_partitions()}")
        job_id = generate_job_id(self.__session_id, tag=tag)
        job = ErJob(id=job_id,
                    name=tag,
                    inputs=[self.ctx.populate_processor(rp.get_store()) for rp in [self] + others],
                    functors=[ErFunctor(name=tag, serdes=SerdesTypes.CLOUD_PICKLE, body=cloudpickle.dumps(func))],
                    options=options)
        args = list()
        for i in range(total_partitions):
            partition_self = job._inputs[0]._partitions[i]
            task = ErTask(id=generate_task_id(job_id, i),
                          name=job._name,
                          inputs=[store._partitions[i] for store in job._inputs],
                          job=job)
            args.append(([task], partition_self._processor._command_endpoint))

        futures = self.__command_client.async_call(
            args=args,
            output_types=[ErPair],
            command_uri=CommandURI(self.RUN_TASK_URI))
        if merge_func is not None:
            results = Queue()
            for fut in futures:
                fut.add_done_callback(lambda x: results.put(x.reuslt()))
            return merge_func(results.get() for _ in range(len(futures)))
        else:
            result = list()
            for future in futures:
                ret_pair = future.result()[0]
                result.append((self.functor_serdes.deserialize(ret_pair._key),
                               self.functor_serdes.deserialize(ret_pair._value)))
            return result


class RollTensor:
    def __init__(self):
        self.rf = None

    def _bin_op(self, other, op):
        def _inner_bin_func(parts):
            in_part, out_part = parts[0], parts[1]
            with _create_store_adapter(in_part) as in_store, _create_store_adapter(out_part) as out_store:
                x = x_store... numpy
                y = y_store.. numpy
                return op(x, y)
        self.rf.with_store(other.rf.store, func=xx)

    def _add(self, other):
        return self._bin_op(other, operator.add)