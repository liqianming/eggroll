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
import unittest
import pyarrow as pa
import numpy as np
import pandas as pd

from eggroll.core.conf_keys import SessionConfKeys, TransferConfKeys, NodeManagerConfKeys, ClusterManagerConfKeys
from eggroll.core.constants import ProcessorTypes, ProcessorStatus
from eggroll.core.meta_model import ErProcessor, ErEndpoint
from eggroll.core.session import ErSession
from eggroll.roll_frame import FrameBatch, TensorBatch
from eggroll.roll_frame.roll_frame import RollFrameContext


def get_test_rf_ctx(is_standalone=False, is_debug=False,
                    manager_port=4670, egg_port=20001, transfer_port=20002, session_id='testing'):
    manager_port = manager_port
    egg_ports = [egg_port]
    egg_transfer_ports = [transfer_port]
    self_server_node_id = 2

    options = {}
    if is_standalone:
        options[SessionConfKeys.CONFKEY_SESSION_DEPLOY_MODE] = "standalone"
    options[TransferConfKeys.CONFKEY_TRANSFER_SERVICE_HOST] = "127.0.0.1"
    options[TransferConfKeys.CONFKEY_TRANSFER_SERVICE_PORT] = str(transfer_port)
    options[ClusterManagerConfKeys.CONFKEY_CLUSTER_MANAGER_PORT] = str(manager_port)
    options[NodeManagerConfKeys.CONFKEY_NODE_MANAGER_PORT] = str(manager_port)
    if is_debug:
        egg = ErProcessor(id=1,
                          server_node_id=self_server_node_id,
                          processor_type=ProcessorTypes.EGG_PAIR,
                          status=ProcessorStatus.RUNNING,
                          command_endpoint=ErEndpoint("127.0.0.1", egg_ports[0]),
                          transfer_endpoint=ErEndpoint("127.0.0.1", egg_transfer_ports[0]))

        roll = ErProcessor(id=1,
                           server_node_id=self_server_node_id,
                           processor_type=ProcessorTypes.ROLL_PAIR_MASTER,
                           status=ProcessorStatus.RUNNING,
                           command_endpoint=ErEndpoint("127.0.0.1", manager_port))
        processors = [egg, roll]
    else:
        processors = None
    session = ErSession(session_id, processors=processors, options=options)
    context = RollFrameContext(session)
    return context


class TestRollFrame(unittest.TestCase):
    def test_tmp(self):
        ctx = get_test_rf_ctx(is_debug=True)
        ctx.load("ns1", "name1")