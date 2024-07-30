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

from .base import NEAlgorithm
from .base import QualityDiversityMethod
from .cma_wrapper import CMA
from .pgpe import PGPE
from .ars import ARS
from .simple_ga import SimpleGA
from .open_es import OpenES
from .cma_evosax import CMA_ES
from .sep_cma_es import Sep_CMA_ES
from .cma_jax import CMA_ES_JAX
from .map_elites import MAPElites
from .iamalgam import iAMaLGaM
from .fcrfmc import FCRFMC
from .crfmnes import CRFMNES
from .ars_native import ARS_native
from .fpgpec import FPGPEC
from .diversifier import Diversifier

Strategies = {
    "CMA": CMA,
    "PGPE": PGPE,
    "SimpleGA": SimpleGA,
    "ARS": ARS,
    "OpenES": OpenES,
    "CMA_ES": CMA_ES,
    "Sep_CMA_ES": Sep_CMA_ES,
    "CMA_ES_JAX": CMA_ES_JAX,
    "MAPElites": MAPElites,
    "iAMaLGaM": iAMaLGaM,
    "FCRFMC": FCRFMC,
    "CRFMNES": CRFMNES,
    "ARS_native": ARS_native,
    "FPGPEC": FPGPEC,
    "Diversifier": Diversifier,
}

__all__ = [
    "NEAlgorithm",
    "QualityDiversityMethod",
    "CMA",
    "PGPE",
    "ARS",
    "SimpleGA",
    "OpenES",
    "CMA_ES",
    "Sep_CMA_ES",
    "CMA_ES_JAX",
    "MAPElites",
    "iAMaLGaM",
    "FCRFMC",
    "CRFMNES",
    "Strategies",
    "ARS_native",
    "FPGPEC",
    "Diversifier",
]
