import typing as t
from enum import Enum

from pydantic import BaseModel

class ParameterException(Exception):
    "Custom Exception for parameters checking."

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class ConnectException(Exception):
    "Custom Exception for milvus connection."

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)

class FiledDataTypes(str, Enum):
    BOOL = "BOOL"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    STRING = "STRING"
    BINARY_VECTOR = "BINARY_VECTOR"
    FLOAT_VECTOR = "FLOAT_VECTOR"

DataTypeByNum = {
    0: "NONE",
    1: FiledDataTypes.BOOL,
    2: FiledDataTypes.INT8,
    3: FiledDataTypes.INT16,
    4: FiledDataTypes.INT32,
    5: FiledDataTypes.INT64,
    10: FiledDataTypes.FLOAT,
    11: FiledDataTypes.DOUBLE,
    20: FiledDataTypes.STRING,
    100: FiledDataTypes.BINARY_VECTOR,
    101: FiledDataTypes.FLOAT_VECTOR,
    999: "UNKNOWN",
}

class IndexTypes(str, Enum):
    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    RNSG = "RNSG"
    HNSW = "HNSW"
    # NSG = "NSG"
    ANNOY = "ANNOY"
    # RHNSQ_FLAT = "RHNSW_FLAT"
    # RHNSQ_PQ = "RHNSW_PQ"
    # RHNSW_SQ = "RHNSW_SQ"
    # BIN_FLAT = "BIN_FLAT"
    # BIN_IVF_FLAT = "BIN_IVF_FLAT"

class IndexParams(str, Enum):
    nlist = "nlist"
    m = "m"
    nbits = "nbits"
    M = "M"
    efConstruction = "efConstruction"
    n_trees = "n_trees"
    PQM = "PQM"
    out_degree = "out_degree"
    candidate_pool_size = "candidate_pool_size"
    search_length = "search_length"
    knng = "knng"

class SearchParams(str, Enum):
    metric_type = "metric_type"
    nprobe = "nprobe"
    search_length = "search_length"
    ef = "ef"
    search_k = "search_k"

class IndexTypesMapValue(BaseModel):
    index_building_parameters: t.List[IndexParams]
    search_parameters: t.List[SearchParams]

# we no longer use this because we are setting the search types explicitly above
IndexTypesMap = {
    IndexTypes.FLAT: IndexTypesMapValue(index_building_parameters=[], search_parameters=[SearchParams.metric_type]),
    IndexTypes.IVF_FLAT: IndexTypesMapValue(index_building_parameters=[IndexParams.nlist], search_parameters=[SearchParams.nprobe]),
    IndexTypes.IVF_SQ8: IndexTypesMapValue(index_building_parameters=[IndexParams.nlist], search_parameters=[SearchParams.nprobe]),
    IndexTypes.IVF_PQ: IndexTypesMapValue(index_building_parameters=[IndexParams.nlist, IndexParams.m, IndexParams.nbits], search_parameters=[SearchParams.nprobe]),
    IndexTypes.RNSG: IndexTypesMapValue(index_building_parameters=[IndexParams.out_degree, IndexParams.candidate_pool_size, IndexParams.search_length, IndexParams.knng], search_parameters=[SearchParams.search_length]),
    IndexTypes.HNSW: IndexTypesMapValue(index_building_parameters=[IndexParams.M, IndexParams.efConstruction], search_parameters=[SearchParams.ef]),
    IndexTypes.ANNOY: IndexTypesMapValue(index_building_parameters=[IndexParams.n_trees], search_parameters=[SearchParams.search_k]),
}

class MetricTypes(str, Enum):
    L2 = "L2"
    IP = "IP"
    HAMMING = "HAMMING"
    TANIMOTO = "TANIMOTO"

class Operators(str, Enum):
    LESS = "<"
    LESS_OR_EQUAL = "<="
    GREATER = ">"
    GREATER_OR_EQUAL = ">="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN = "in"
