### Hexagon runtime tests

This directory sets up Hexagon unit tests using gtest for the Hexagon runtime code. Since the current hexagon runtime uses run_main_on_hexagon and to execute on target, this setup just creates a file called `run_tests.sh`, which runs all the tests implemented here.

This script expects the following environment variables to be set to run on target:

1. $ANDROID_HOST
2. $ANDROID_SERIAL

#### How it works?

The main file that starts the gtest invocation is `hexagon_runtime_test.cpp`. This defines the entry point (`main` function) and does some pre-processing before invoking gtest with appropriate arguments. To pass arguments to gtest itself (like `--gtest-filter=*foo*`), the main funtion accepts an argument (`--gtest-args=""`), where everything inside the quotes ("") is passed to gtest as arguments to gtest invocation.

For example:

an invocation like `./run_main_on_hexagon 3 libhexagon_runtime_test.so --gtest-args="--gtest-filter=*VTCM* --gtest-repeat=3"` would pass the filter and repeat arguments to the gtest invocation and hence only tests that contain the name "VTCM" would be exectued and they would be repeated 3 times.

#### Executing tests

The tests can be executed by running `./run_tests.sh` from the build directory. This file is regenerated everytime the project is rebuilt, and any `--gtest-args` to be passed can be passed by editing the call the `adb shell` inside the script.
TODO: modify the generated script to accept gtest-args as a command line argument and pass it to adb shell
