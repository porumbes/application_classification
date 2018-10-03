### application_classification

Implementation of `ApplicationClassification` workflow. 

Produces identical results to [PNNL code](https://gitlab.hiveprogram.com/pnnl/ApplicationClassification/) on `georgiy` and `rmat` example datasets.

#### TODO

- [ ] Optimize performance
 - Fuse kernels
 - Get rid of transposes
 - Segmented reduce instead of reduce by key?
 - Could probably improve memory usage
- [ ] Profiling
- [ ] _optional_ more correctness checking