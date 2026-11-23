//===- HMXPower.h - HMX power vote helper ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Header-only so runtime modules can use it without a link dependency.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HMXPOWER_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_HMXPOWER_H_

#include <string.h>

#include "HAP_power.h"

// Powers up HMX. Votes the core clock first: an HMX-only vote drops the core
// clock (seen on v75/v81), and HMX follows the core clock. Same as HexKL.
// `powerContext` must outlive the vote.
inline int hexagonPowerUpHmx(void *powerContext) {
  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_DCVS_v3;
  request.dcvs_v3.set_dcvs_enable = TRUE;
  request.dcvs_v3.dcvs_enable = TRUE;
  request.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  request.dcvs_v3.set_core_params = TRUE;
  request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.set_bus_params = TRUE;
  request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
  int ret = HAP_power_set(powerContext, &request);
  if (ret != AEE_SUCCESS)
    return ret;

  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = TRUE;
  return HAP_power_set(powerContext, &request);
}

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HMXPOWER_H_
