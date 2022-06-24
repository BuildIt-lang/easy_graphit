# GraphIt DSL with BuildIt 

This is an implementation of the [GraphIt DSL](https://graphit-lang.org) and its GPU backend with BuildIt. This implementation includes the language API (algorithm and scheduling), analysis and specialization and GPU code generation. 

Details of all the techniques to implement analysis, scheduling and GPU code generation are explained in our paper [GraphIt to CUDA compiler in 2021 LOC: A case for high-performance DSL implementation via staging with BuilDSL](https://buildit.so/publications) [1]. 

Steps for building this repo, running all the evaluations and comparing results with GraphIt's own implementation are in this [companion repo](https://github.com/BuildIt-lang/buildsl_cgo22_artifacts). The companion repo also has a tutorial for building a new DSL with the techniques in this work. 



1 - A. Brahmakshatriya and S. Amarasinghe, "GraphIt to CUDA Compiler in 2021 LOC: A Case for High-Performance DSL Implementation via Staging with BuilDSL," 2022 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), 2022, pp. 53-65, doi: 10.1109/CGO53902.2022.9741280.
