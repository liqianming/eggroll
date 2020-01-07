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

import unittest

from eggroll.core.session import ErSession
from eggroll.roll_pair.roll_pair import RollPair, RollPairContext
from eggroll.roll_pair.test.roll_pair_test_assets import get_debug_test_context
from eggroll.roll_site.roll_site import RollSiteContext

is_standalone = False
manager_port_guest = 4671
egg_port_guest = 20003
transfer_port_guest = 20004
manager_port_host = 4670
egg_port_host = 20001
transfer_port_host = 20002
remote_parties = [('host', '10001')]
get_parties = [('guest', '10002')]

options_host = {'runtime_conf_path': 'conf/role_conf.host.json',
                'server_conf_path': 'conf/server_conf.host.json'}
options_guest = {'runtime_conf_path': 'conf/role_conf.guest.json',
                 'server_conf_path': 'conf/server_conf.guest.json'}

class TestRollSite(unittest.TestCase):
    def test_host_init(self):
        rp_context = get_debug_test_context(is_standalone, manager_port_host, egg_port_host, transfer_port_host, 'testing')
        RollSiteContext("atest", options=options_host, rp_ctx=rp_context)

    def test_remote(self):
        rp_context = get_debug_test_context(is_standalone, manager_port_guest, egg_port_guest, transfer_port_guest, 'testing_guest')
        context = RollSiteContext("atest", options=options_guest, rp_ctx=rp_context)
        _tag = "Hello2"
        rs = context.load(name="RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag))
        fp = open("testA.model", 'r')
        obj = fp.read(35)
        futures = rs.push(obj, remote_parties)
        fp.close()
        for future in futures:
            role, party = future.result()
            print("result:", role, party)

    def test_get(self):
        rp_context = get_debug_test_context(is_standalone, manager_port_host, egg_port_host, transfer_port_host, 'testing')
        context = RollSiteContext("atest", options=options_host, rp_ctx=rp_context)
        _tag = "Hello2"
        rs = context.load(name="RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag))
        futures = rs.pull(get_parties)
        for future in futures:
            obj = future.result()
            if isinstance(obj, RollPair):
                key = 'hello'
                print("obj:", obj.get(key))
            else:
                print("obj:", obj)

    def test_host_init_rollpair(self):
        rp_context = get_debug_test_context(is_standalone, manager_port_host, egg_port_host, transfer_port_host, 'testing')
        RollSiteContext("atest2", options=options_host, rp_ctx=rp_context)

    def test_remote_rollpair(self):
        rp_context = get_debug_test_context(is_standalone, manager_port_guest, egg_port_guest, transfer_port_guest, 'testing_guest')
        data = [("k1", "v1"), ("k2", "v2"), ("k3", "v3")]
        context = RollSiteContext("atest2", options=options_guest, rp_ctx=rp_context)
        rp_options = {'include_key': True}
        rp = rp_context.load("namespace", "name").put_all(data, options=rp_options)
        _tag = "Hello"
        rs = context.load(name="roll_pair_name.table", tag="roll_pair_tag")
        futures = rs.push(rp, remote_parties)
        for future in futures:
            role, party = future.result()
            print("result:", role, party)

    def test_get_rollpair(self):
        rp_context = get_debug_test_context(is_standalone, manager_port_host, egg_port_host, transfer_port_host, 'testing')
        context = RollSiteContext("atest2", options=options_host, rp_ctx=rp_context)
        _tag = "roll_pair_tag"
        rs = context.load(name="roll_pair_name.table", tag="{}".format(_tag))
        futures = rs.pull(get_parties)
        for future in futures:
            obj = future.result()
            if isinstance(obj, RollPair):
                key = "k1"
                print("obj:", obj.get(key))
            else:
                print("obj:", obj)

    def test_get_table(self):
        session = ErSession(session_id='testing')
        rpc = RollPairContext(session)

        rp = rpc.load('atest2', '__federation__-atest2-roll_pair_name.table-roll_pair_tag-guest-10002-host-10001')

        print(f'k1: {rp.get("k1")}')

    def test_get_all(self):
        session = ErSession(session_id='testing')
        rpc = RollPairContext(session)

        rp = rpc.load('atest2', '__federation__-atest2-roll_pair_name.table-roll_pair_tag-guest-10002-host-10001')
        print(list(rp.get_all()))