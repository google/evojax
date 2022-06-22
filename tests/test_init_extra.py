# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEST_EVOSAX = False


class TestTask:
    def test_mnist(self):
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor <= 9:
            # python<=3.9, required by the optional torchvision (see https://pypi.org/project/torchvision/)
            from evojax.task.mnist import MNIST
            _ = MNIST()
            assert True

    def test_mdkp(self):
        from evojax.task.mdkp import MDKP
        _ = MDKP()
        assert True

    def test_procgen(self):
        from evojax.task.procgen_task import ProcgenTask
        _ = ProcgenTask(env_name='starpilot')
        assert True


class TestPolicy:
    pass


class TestAlgo:
    def test_cma(self):
        from evojax.algo import CMA
        _ = CMA(pop_size=16, param_size=16)
        assert True

    def test_simple_ga(self):
        from evojax.algo import SimpleGA
        _ = SimpleGA(pop_size=16, param_size=16)
        assert True

    if TEST_EVOSAX:
        def test_open_es(self):
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                # python>=3.7, required by the optional evosax
                from evojax.algo import OpenES
                _ = OpenES(pop_size=16, param_size=16)
                assert True

        def test_ars(self):
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                # python>=3.7, required by the optional evosax
                from evojax.algo import ARS
                _ = ARS(pop_size=16, param_size=16)
                assert True

        def test_iamalgam(self):
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                # python>=3.7, required by the optional evosax
                from evojax.algo import iAMaLGaM
                _ = iAMaLGaM(pop_size=16, param_size=16)
                assert True
