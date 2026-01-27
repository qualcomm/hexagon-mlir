//===- UserDMARegisters.h - Programming Hexagon User DMA registers --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Reference: Qualcomm Hexagon User DMA Specification 80-V9418-29 Rev. D
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBACKEND_BIN_RUNTIME_USERDMA_REGISTERS_H
#define HEXAGONBACKEND_BIN_RUNTIME_USERDMA_REGISTERS_H
namespace hexagon {
namespace userdma {
/* Register offset */
#define regDM0 0x0 // per engine configuration
#define regDM1 0x1 // reserved
#define regDM2 0x2 // global control shared by all DMA Engines
#define regDM3 0x3 // reserved
#define regDM4 0x4 // global error syndrome register shared by all DMA Engines
#define regDM5 0x5 // global error syndrome register shared by all DMA Engines

// DM0[1:0]
#define DM0_STATUS_MASK 0x00000003
#define DM0_STATUS_SHIFT 0
#define DM0_STATUS_IDLE 0
#define DM0_STATUS_RUN 1
#define DM0_STATUS_ERROR 2

// DM0[31:4]
// Descriptors addresses must be (minimum) 16 byte aligned
// -> Lower 4 bits masked to clear DMA Status
// -> But, descriptor address is not shifted
#define DM0_DESC_ADDR_MASK 0xFFFFFFF0
#define DM0_DESC_ADDR_SHIFT 0

// DM2[0]
#define DM2_GUEST_MODE_STALL_MASK 0x00000001
#define DM2_GUEST_MODE_STALL_SHIFT 0
#define DM2_GUEST_MODE_STALL_YES 0
#define DM2_GUEST_MODE_STALL_NO 1

// DM2[1]
#define DM2_MONITOR_MODE_STALL_MASK 0x00000002
#define DM2_MONITOR_MODE_STALL_SHIFT 1
#define DM2_MONITOR_MODE_STALL_YES 0
#define DM2_MONITOR_MODE_STALL_NO 1

// DM2[3]
#define DM2_EXCEPTION_MODE_CONTINUE_MASK 0x00000008
#define DM2_EXCEPTION_MODE_CONTINUE_SHIFT 3
#define DM2_EXCEPTION_MODE_CONTINUE_YES 0
#define DM2_EXCEPTION_MODE_CONTINUE_NO 1

// DM2[4]
#define DM2_DEBUG_MODE_CONTINUE_MASK 0x00000010
#define DM2_DEBUG_MODE_CONTINUE_SHIFT 4
#define DM2_DEBUG_MODE_CONTINUE_NO 0
#define DM2_DEBUG_MODE_CONTINUE_YES 1

// DM2[6:5]
#define DM2_TRAFFIC_PRIORITY_MASK 0x00000060
#define DM2_TRAFFIC_PRIORITY_SHIFT 5
#define DM2_TRAFFIC_PRIORITY_IDLE 0
#define DM2_TRAFFIC_PRIORITY_LOW 1
#define DM2_TRAFFIC_PRIORITY_INHERIT 2
#define DM2_TRAFFIC_PRIORITY_HIGH 3

// DM2[7]
#define DM2_DLBC_ENABLE_MASK 0x00000080
#define DM2_DLBC_ENABLE_SHIFT 7
#define DM2_DLBC_DISABLE 0
#define DM2_DLBC_ENABLE 1

// DM2[8]
#define DM2_OOO_WRITE_MASK 0x00000100
#define DM2_OOO_WRITE_SHIFT 8
#define DM2_OOO_WRITE_ENABLE 0
#define DM2_OOO_WRITE_DISABLE 1

// DM2[9]
#define DM2_ERROR_EXCEPTION_MASK 0x00000200
#define DM2_ERROR_EXCEPTION_SHIFT 9
#define DM2_ERROR_EXCEPTION_GENERATE_NO 0
#define DM2_ERROR_EXCEPTION_GENERATE_YES 1

// DM2[23:16]
#define DM2_OUTSTANDING_READ_MASK 0x00FF0000
#define DM2_OUTSTANDING_READ_SHIFT 16

// DM2[31:24]
#define DM2_OUTSTANDING_WRITE_MASK 0xFF000000
#define DM2_OUTSTANDING_WRITE_SHIFT 24

// DM4[0]
#define DM4_ERROR_MASK 0x00000001
#define DM4_ERROR_SHIFT 0
#define DM4_ERROR_NO 0
#define DM4_ERROR_YES 1

// DM4[7:4]
#define DM4_THREAD_ID_MASK 0x000000F0
#define DM4_THREAD_ID_SHIFT 4

// DM4[15:8]
#define DM4_SYNDRONE_CODE_MASK 0x0000FF00
#define DM4_SYNDRONE_CODE_SHIFT 8
#define DM4_SYNDRONE_CODE_DM_COMMAND_ERROR 0
#define DM4_SYNDRONE_CODE_DESCRIPTOR_INVALID_ALIGNMENT 1
#define DM4_SYNDRONE_CODE_DESCRIPTOR_INVALID_TYPE 2
#define DM4_SYNDRONE_CODE_UNSUPPORTED_ADDRESS 3
#define DM4_SYNDRONE_CODE_UNSUPPORTED_BYPASS_MODE 4
#define DM4_SYNDRONE_CODE_UNSUPPORTED_COMP_FORMAT 5
#define DM4_SYNDRONE_CODE_DESCRIPTOR_ROI_ERROR 6
#define DM4_SYNDRONE_CODE_BUS_ERROR_DESCRIPTOR_RW 7
#define DM4_SYNDRONE_CODE_BUS_ERROR_L2_READ 8
#define DM4_SYNDRONE_CODE_BUS_ERROR_L2_WRITE 9
// TODO: Bus Error (10) on Compression Metadata?
// TODO: Definition? Not in the spec.
#define DM4_SYNDRONE_CODE_INVALID_ACCESS_RIGHTS 102
#define DM4_SYNDRONE_CODE_DATA_TIMEOUT 103
#define DM4_SYNDRONE_CODE_DATA_ABORT 104

// DM5
#define DM5_SYNDRONE_ADDR_MASK 0xFFFFFFFF
#define DM5_SYNDRONE_ADDR_SHIFT 0

// DM2[0]
static inline unsigned int dm2GetGuestMode(unsigned int cfg) {
  return (cfg & DM2_GUEST_MODE_STALL_MASK) >> DM2_GUEST_MODE_STALL_SHIFT;
}

// DM2[0]
static inline void dm2SetGuestMode(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_GUEST_MODE_STALL_MASK;
  *cfg |= ((v << DM2_GUEST_MODE_STALL_SHIFT) & DM2_GUEST_MODE_STALL_MASK);
}

// DM2[1]
static inline unsigned int dm2GetMonitorMode(unsigned int cfg) {
  return (cfg & DM2_MONITOR_MODE_STALL_MASK) >> DM2_MONITOR_MODE_STALL_SHIFT;
}

// DM2[1]
static inline void dm2SetMonitorMode(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_MONITOR_MODE_STALL_MASK;
  *cfg |= ((v << DM2_MONITOR_MODE_STALL_SHIFT) & DM2_MONITOR_MODE_STALL_MASK);
}

// DM2[3]
static inline unsigned int dm2GetExceptionMode(unsigned int cfg) {
  return (cfg & DM2_EXCEPTION_MODE_CONTINUE_MASK) >>
         DM2_EXCEPTION_MODE_CONTINUE_SHIFT;
}

// DM2[3]
static inline void dm2SetExceptionMode(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_EXCEPTION_MODE_CONTINUE_MASK;
  *cfg |= ((v << DM2_EXCEPTION_MODE_CONTINUE_SHIFT) &
           DM2_EXCEPTION_MODE_CONTINUE_MASK);
}

// DM2[4]
static inline unsigned int dm2GetDebugMode(unsigned int cfg) {
  return (cfg & DM2_DEBUG_MODE_CONTINUE_MASK) >> DM2_DEBUG_MODE_CONTINUE_SHIFT;
}

// DM2[4]
static inline void dm2SetDebugMode(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_DEBUG_MODE_CONTINUE_MASK;
  *cfg |= ((v << DM2_DEBUG_MODE_CONTINUE_SHIFT) & DM2_DEBUG_MODE_CONTINUE_MASK);
}

// DM2[6:5]
static inline unsigned int dm2GetPriority(unsigned int cfg) {
  return (cfg & DM2_TRAFFIC_PRIORITY_MASK) >> DM2_TRAFFIC_PRIORITY_SHIFT;
}

// DM2[6:5]
static inline void dm2SetPriority(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_TRAFFIC_PRIORITY_MASK;
  *cfg |= ((v << DM2_TRAFFIC_PRIORITY_SHIFT) & DM2_TRAFFIC_PRIORITY_MASK);
}

// DM2[7]
static inline unsigned int dm2GetDLBCEnable(unsigned int cfg) {
  return (cfg & DM2_DLBC_ENABLE_MASK) >> DM2_DLBC_ENABLE_SHIFT;
}

// DM2[7]
static inline void dm2SetDLBCEnable(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_DLBC_ENABLE_MASK;
  *cfg |= ((v << DM2_DLBC_ENABLE_SHIFT) & DM2_DLBC_ENABLE_MASK);
}

// DM2[8]
static inline unsigned int dm2GetOOOWriteCtrl(unsigned int cfg) {
  return (cfg & DM2_OOO_WRITE_MASK) >> DM2_OOO_WRITE_SHIFT;
}

// DM2[8]
static inline void dm2SetOOOWriteCtrl(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_OOO_WRITE_MASK;
  *cfg |= ((v << DM2_OOO_WRITE_SHIFT) & DM2_OOO_WRITE_MASK);
}

// DM2[9]
static inline unsigned int dm2GetErrorExceptionCtrl(unsigned int cfg) {
  return (cfg & DM2_ERROR_EXCEPTION_MASK) >> DM2_ERROR_EXCEPTION_SHIFT;
}

// DM2[9]
static inline void dm2SetErrorExceptionCtrl(unsigned int *cfg, unsigned int v) {
  *cfg &= ~DM2_ERROR_EXCEPTION_MASK;
  *cfg |= ((v << DM2_ERROR_EXCEPTION_SHIFT) & DM2_ERROR_EXCEPTION_MASK);
}

// DM2[23:16]
static inline unsigned int
dm2_getOutstandingTransactionsRead(unsigned int cfg) {
  return (cfg & DM2_OUTSTANDING_READ_MASK) >> DM2_OUTSTANDING_READ_SHIFT;
}

// DM2[23:16]
static inline void dm2SetOutstandingTransactionsRead(unsigned int *cfg,
                                                     unsigned int v) {
  *cfg &= ~DM2_OUTSTANDING_READ_MASK;
  *cfg |= ((v << DM2_OUTSTANDING_READ_SHIFT) & DM2_OUTSTANDING_READ_MASK);
}

// DM2[31:24]
static inline unsigned int
dm2GetOutstandingTransactionsWrite(unsigned int cfg) {
  return (cfg & DM2_OUTSTANDING_WRITE_MASK) >> DM2_OUTSTANDING_WRITE_SHIFT;
}

// DM2[31:24]
static inline void dm2SetOutstandingTransactionsWrite(unsigned int *cfg,
                                                      unsigned int v) {
  *cfg &= ~DM2_OUTSTANDING_WRITE_MASK;
  *cfg |= ((v << DM2_OUTSTANDING_WRITE_SHIFT) & DM2_OUTSTANDING_WRITE_MASK);
}

/*--------------------------------------------------------------------------*/

// DM4[0]
static inline unsigned int dm4GetError(unsigned int cfg) {
  return (cfg & DM4_ERROR_MASK) >> DM4_ERROR_SHIFT;
}

// DM4[7:4]
static inline unsigned int dm4GetEngineID(unsigned int cfg) {
  return (cfg & DM4_THREAD_ID_MASK) >> DM4_THREAD_ID_SHIFT;
}

// DM4[15:8]
static inline unsigned int dm4GetSyndroneCode(unsigned int cfg) {
  return (cfg & DM4_SYNDRONE_CODE_MASK) >> DM4_SYNDRONE_CODE_SHIFT;
}

/*--------------------------------------------------------------------------*/

// DM5
static inline unsigned int dm5GetSyndroneAddr(unsigned int cfg) {
  return (cfg & DM5_SYNDRONE_ADDR_MASK) >> DM5_SYNDRONE_ADDR_SHIFT;
}
} // namespace userdma
} // namespace hexagon

#endif // HEXAGONBACKEND_BIN_RUNTIME_USERDMA_REGISTERS_H
