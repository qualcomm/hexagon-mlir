## ETM and PyETM:
- "pyetm is a tool which processes Hexagon ETM traces and derives the flow of each thread of the processor." - To learn more about pyetm, check out go/pyetm

- ## Requirement:
    - To use PyETM, user needs to join the list - https://lists.qualcomm.com/ListManager?id=role.qct.modem-perf.pyetm.users

- ## Notes on CDSP Meta:
    - Device rental (drental) devices use an older meta version, and their CDSP binaries are no longer accessible via go/findit.
    - These binaries are required for PyETM processing.
    - All drental Lanai devices currently use the same meta: "Lanai.LA.1.0-00788-APQ.INT-2" and the corresponding CDSP build of this meta is - "CDSP.HT.3.0-00724.1-LANAI-1". This cdsp build is saved in a local Austin server at: /prj/qct/asw/crmbuilds/nsid-aus-01
    - However, this path is not accessible from Vegas machines.
    - To resolve this, CDSP binaries of this specific cdsp build have been copied over here -> /prj/qct/llvm/target/ml/CDSP.HT.3.0-00724.1-LANAI-1_cdsp-bins/
    - It's been confirmed with Amy that all devices in the default queue should consistently match this meta. She also mentioned that older devices like Lanai are upgraded less frequently, so we can assume this meta will remain stable in the near future.

- ## PyETM result:
    - Results will be saved in the <timestamped_dir>/etm_pyetm/pyetm_results. Look at the debug.txt in that dir. If the value of PercentageProcessed is more than 90%, it can be interpreted as success in decoding the trace file.

- ## Constraints:
    - To use pyetm, the application execution time should not be more than a couple of seconds. If it is more than that, this tool is not recommended to use. If user really needs to use it, they may need to collect multiple trace files and let pyetm analyze them together having multitracing enabled in the config.json file --> "enable_multi_trace_timeline" : "True"