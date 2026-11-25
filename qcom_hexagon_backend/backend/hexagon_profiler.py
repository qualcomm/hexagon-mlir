# ===- hexagon_profiler.py --------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import json
import shlex
import subprocess
import time
from pathlib import Path

from triton.backends.qcom_hexagon_backend import knobs


class HexagonProfiler:
    """Hexagon Profiler"""

    _SYSMON_ELF_NAME_MAXLEN = 30

    def __init__(
        self,
        etm_local_dir: Path,
        profiling_mode=None,
        device_lib_path="",
        local_lib_filename="",
    ):
        if profiling_mode not in ["etm"]:
            raise RuntimeError("Profiling mode was not set or was not a valid one.")

        # Validate required environment variables
        required_vars = {
            "ANDROID_HOST": knobs.hexagon.android_host,
            "ANDROID_SERIAL": knobs.hexagon.android_serial,
            "HEXAGON_ARCH_VERSION": knobs.hexagon.arch_version,
            "PYETM": knobs.hexagon_profiler.pyetm,
            "CDSP_PATH": knobs.hexagon_profiler.cdsp_path,
            "HEXAGON_SDK_ROOT": knobs.hexagon.sdk_root,
        }
        missing_vars = [name for name, value in required_vars.items() if value is None]
        if missing_vars:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please source local_env.sh before running."
            )

        self.ANDROID_HOST = knobs.hexagon.android_host
        self.ANDROID_SERIAL = knobs.hexagon.android_serial
        self.Q6_VERSION = knobs.hexagon.arch_version
        self.PYETM = str(knobs.hexagon_profiler.pyetm)
        self.CDSP_PATH = str(knobs.hexagon_profiler.cdsp_path)
        self.HEXAGON_SDK_ROOT = str(knobs.hexagon.sdk_root)

        self.ETM_FILES_PATH = etm_local_dir / "etm_files"
        self.APP_BINS = etm_local_dir / "app_bins"
        self.PYETM_RESULTS = etm_local_dir / "pyetm_results"
        self._profiling_mode = profiling_mode
        self.reboot_device()
        self.enable_etm_trace_ca()
        self.etm_prep()
        self.device_lib_path = device_lib_path
        self.local_lib_filename = local_lib_filename

    def run_bash_command(
        self, command, adb: bool = True, capture_output=False, text=False
    ):
        try:
            if adb:
                cmd = ["adb", "-H", self.ANDROID_HOST, "-s", self.ANDROID_SERIAL,
                       *shlex.split(command)]
            else:
                cmd = shlex.split(command)
            print(f"Running: {shlex.join(cmd)}", flush=True)
            return subprocess.run(
                cmd,
                check=True,
                capture_output=capture_output,
                text=text,
            )
        except subprocess.CalledProcessError as e:
            print(f"==> Error executing command: {full_command}")
            print(f"==> Return code: {e.returncode}")
            if e.output:
                print(f"==> Output: {e.output}")
            raise

    # Device is rebooted.
    def reboot_device(self):
        commands = [
            "reboot",
            "wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done; input keyevent 82'",
            "wait-for-device root",
            "wait-for-device remount",
        ]
        print("==> Device is rebooting")
        for cmd in commands:
            time.sleep(5)
            self.run_bash_command(cmd)

        print("==> Device is back in root mode")

    # Enabling CDSP ETM Trace.
    def enable_etm_trace_ca(self):
        commands = [
            'shell "echo 0x1f > /vendor/lib/rfsa/adsp/cdsprpcd.farf"',
            f"push {self.HEXAGON_SDK_ROOT}/tools/utils/sysmon/sysMonApp /data/local/tmp",
            'shell "chmod 777 /data/local/tmp/sysMonApp"',
            'shell "echo 1 > /sys/bus/coresight/reset_source_sink"',
            'shell "echo 0 > /sys/bus/coresight/devices/coresight-stm/enable_source"',
            'shell "echo 0 > /sys/bus/coresight/devices/coresight-tmc-etr/enable_sink"',
            'shell "echo 0x10000000 > /sys/bus/coresight/devices/coresight-tmc-etr/buffer_size"',
            'shell "echo 0 > /sys/bus/coresight/devices/coresight-stm/hwevent_enable"',
            'shell "echo mem > /sys/bus/coresight/devices/coresight-tmc-etr/out_mode"',
            'shell "echo 1 > /sys/bus/coresight/devices/coresight-tmc-etr/enable_sink"',
            'shell "echo 1 > /sys/bus/coresight/devices/coresight-turing-etm0/enable_source"',
            'shell "/data/local/tmp/sysMonApp etmTrace --command etm --q6 CDSP --etmType ca_pc"',
            "shell \"/vendor/bin/qdss_qmi_helper cdsp etm_config 'dsp mode 0x9'\"",
        ]

        for cmd in commands:
            self.run_bash_command(cmd)

    # Pull the cdsp bins from the device.
    def etm_prep(self):
        cmd = f"mkdir -p {self.ETM_FILES_PATH}"
        self.run_bash_command(cmd, adb=False)

        commands = [
            f"pull /vendor/dsp/cdsp/fastrpc_shell_unsigned_3 {self.ETM_FILES_PATH}",
            f"pull /vendor/dsp/cdsp/libc++abi.so.1 {self.ETM_FILES_PATH}",
            f"pull /vendor/dsp/cdsp/libc++.so.1 {self.ETM_FILES_PATH}",
            f'shell find /vendor/firmware_mnt/image/ -name "cdsp.*" -print0 | xargs -i -0 -n 1 adb -H {self.ANDROID_HOST} -s {self.ANDROID_SERIAL} pull {{}} {self.ETM_FILES_PATH}',
            'shell "mkdir -p /data/local/traces; rm -rf /data/local/traces/*"',
            'shell "rm -rf /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf"',
            'shell "touch /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf"',
            'shell "echo 0x1f > /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf"',
            "logcat -c",
            "logcat -G 16M",
        ]

        for cmd in commands:
            self.run_bash_command(cmd)

    # extract DDR traces from RAM dump to a file.
    def dump_etm_trace(self):
        commands = [
            'shell "cat /dev/coresight-tmc-etr > /data/local/traces/trace.bin"',
            f"logcat -d > {self.ETM_FILES_PATH}/logcat.log 2>&1",
            f"shell /data/local/tmp/sysMonApp etmTrace --command dll --q6 cdsp > {self.ETM_FILES_PATH}/sysmon.txt 2>&1",
            f"wait-for-device pull /data/local/traces/trace.bin {self.ETM_FILES_PATH}",
        ]

        for cmd in commands:
            self.run_bash_command(cmd)

    # Get the load address for necessary elfs.
    def get_load_addresses(self):
        def run_command(lib_name):
            cmd = f"grep -A 5 -m 1 \"{lib_name}\" {self.ETM_FILES_PATH}/sysmon.txt | grep \"data.LOAD_ADDRESS\" | awk -F ' = ' '{{print $2}}'"
            result = self.run_bash_command(
                cmd, adb=False, capture_output=True, text=True
            )
            return result.stdout.strip()

        addresses = {
            "APP_SO_LOAD_ADDRESS": run_command(
                self.device_lib_path[: self._SYSMON_ELF_NAME_MAXLEN]
            ),
            "librun_main_on_hexagon_skel": run_command("librun_main_on_hexagon_skel"),
            "fastrpc_shell_3_la": run_command("fastrpc_shell_unsigned_3"),
            "libcpp_abi_la": run_command("libc++abi.so.1"),
            "libcpp_la": run_command("libc++.so.1"),
        }

        return addresses

    # Create config.json file needed for pyetm.
    def write_config_lanai(self, addresses):
        config_path = self.ETM_FILES_PATH / "config.json"

        config = {
            "elf_list": [
                f"{self.ETM_FILES_PATH}/cdsp.mdt",
                f"{self.APP_BINS}/{self.local_lib_filename}",
                f"{self.APP_BINS}/librun_main_on_hexagon_skel.so",
                f"{self.ETM_FILES_PATH}/fastrpc_shell_unsigned_3",
                f"{self.ETM_FILES_PATH}/libc++abi.so.1",
                f"{self.APP_BINS}/libc++.so.1",
                f"{self.CDSP_PATH}/ROOT_lanai.cdsp.prodQ.elf",
                f"{self.CDSP_PATH}/SECURE_PD_USER_lanai.cdsp.prodQ.elf",
                f"{self.CDSP_PATH}/SRM_IMG_lanai.cdsp.prodQ.elf",
                f"{self.CDSP_PATH}/UKERNEL_lanai.cdsp.prodQ.elf",
            ],
            "atid": [38, 39],
            "elf_offsets": [
                "0x0",
                addresses["APP_SO_LOAD_ADDRESS"],
                addresses["librun_main_on_hexagon_skel"],
                addresses["fastrpc_shell_3_la"],
                addresses["libcpp_abi_la"],
                addresses["libcpp_la"],
                "0x0",
                "0x0",
                "0x0",
                "0x0",
            ],
            "ver": "v75",
            "target": "Hexagon",
            "llvm_flag": "--mattr=hmx,hlx,hvx",
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    # Run tool pypyetm to analyze the etm traces and create report.
    def run_pyetm(self):
        if not self.PYETM.exists():
            raise RuntimeError(
                f"==> Path to pyetm tool does not exist: {self.PYETM}. "
                "Run pyetm manually using the traces collected from this run."
            )

        commands = [
            f"rm -rf {self.PYETM_RESULTS}/*",
            f"{self.PYETM} {self.ETM_FILES_PATH}/config.json {self.PYETM_RESULTS} {self.ETM_FILES_PATH}/trace.bin > {self.ETM_FILES_PATH}/pypyetm.log 2>&1",
        ]

        for cmd in commands:
            self.run_bash_command(cmd, adb=False)

        print(f"==> PyETM analysis ended and reports are saved in {self.PYETM_RESULTS}")

    def check_cdsp_version(self):
        if not self.CDSP_PATH.exists():
            raise RuntimeError(
                f"==> CDSP bins path does not exist: {self.CDSP_PATH}. "
                "Run pyetm manually using the traces collected from this run."
            )

        cmd = "shell \"cd /vendor/firmware_mnt/verinfo; cat ver_info.txt\" | grep cdsp | awk -F ': ' '{print $2}' | tr -d '\",'"

        result = self.run_bash_command(cmd, capture_output=True, text=True)
        cdsp_version = result.stdout.strip()

        expected = "CDSP.HT.3.0-00953-LANAI-1"
        if cdsp_version == expected:
            print(f"==> CDSP bins are used from {self.CDSP_PATH}")
        else:
            raise RuntimeError(
                f"==> Error: CDSP bins not available for version '{cdsp_version}'. "
                "Run pyetm manually using the traces collected from this run."
            )

    def analyze_trace(self):
        self.dump_etm_trace()
        self.check_cdsp_version()
        addresses = self.get_load_addresses()
        assert (
            self.Q6_VERSION == "75"
        ), "Only support profiling using lanai at the moment"
        self.write_config_lanai(addresses)
        self.run_pyetm()
