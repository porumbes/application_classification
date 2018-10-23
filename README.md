### application_classification

Implementation of `ApplicationClassification` workflow. 

Produces identical results to [PNNL code](https://gitlab.hiveprogram.com/pnnl/ApplicationClassification/) on `georgiy` and `rmat` example datasets.

__Note: This is an adaptation of the master branch to reduce excessive use of the `__transpose` kernel.  It is faster (30% faster on the RMAT test graph).  However, it has been tested less thoroughly, and some of the variable names (`num_rows`, `num_cols`, etc) are incorrect because we've transposed the data matrices.__

#### TODO

- [ ] Optimize performance
 - Fuse kernels
 - Segmented reduce instead of reduce by key?
 - Could probably improve memory usage
- [ ] Profiling
- [ ] _optional_ more correctness checking