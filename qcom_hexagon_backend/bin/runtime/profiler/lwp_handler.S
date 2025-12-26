// ===- lwp_handler.S --------------------------------------------------------===
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
// ===------------------------------------------------------------------------===
.globl lwp_handler
.falign
.type   lwp_handler,@function
.hidden lwp_counter
.hidden lwp_buffer
.hidden lwp_buffer_size
.hidden lwp_buffer_count

lwp_handler:
  { 
    allocframe(#32)                             // Allocate 32 bytes on the stack to save R0-R5 registers (6*4bytes) and P0-P3 (4*1byte) + 4 unused bytes as the stack has to be 8-bytes aligned
    memd(r29+#-16) = r5:4                       // Save R5,R4
    r5 = p3:0                                   // We will save P3:0 but we need an intermediate usual register (R5) that has already been saved
  }
  {
    memd(r29+#16) = r3:2                        // Save R3,R2
    memd(r29+#8) = r1:0                         // Save R1, R0
  }
  {
    memw(r29+#0) = r5                           // Save P3:0 (via R5)
  }
  {
    r5 = add(pc,##lwp_counter@PCREL)            // Compute the absolute address of 'lwp_counter' via PC offset
  }
  {
    r3 = add(pc,##lwp_buffer_count@PCREL)       // Compute the absolute address of 'lwp_buffer_count' via PC offset
  }
  {
    r3 = memw(r3+#0)                            // Get the lwp_buffer_count value (offset into the main buffer)
  }
  {
    r4 = memw(r5+r0<<#2)                        // Get the handler invocation count for the ID (passed in R0)
    r1 = add(pc,##lwp_buffer_size@PCREL)        // Compute the absolute address of 'lwp_buffer_size' via PC offset
  }
  {
    r4 = add(r4,#1)                             // Increment count
    memw(r5+r0<<#2) = r4.new                    // Update count in lwp_counter for a given ID
    r1 = memw(r1+#0)                            // Get the buffer size
  }
  {
    p0 = cmp.gtu(r4,#200)                       // Exit if count for a given ID is greater than 200
    if (p0.new) jump:nt .LBB0_3
    r5 = add(pc,##lwp_buffer@PCREL)             // Compute the absolute address of 'lwp_buffer'
  }
  {
    r2 = add(pc,##lwp_buffer_count@PCREL)       // Compute the absolute address of 'lwp_buffer_count' via PC offset
  }
  {
    r4 = add(r3,#3)                             // Increment the offset by 3 since 3 int32 values are stored for each invocation
    if (!cmp.gtu(r1,r4.new)) jump:t .LBB0_3     // Exit if the main lwp buffer has run out of space
  }
  {
    r5 = addasl(r5,r3,#2)                       // Get the address where the data needs to be recorded
    memw(r2+#0) = r4                            // Save next offset into lwp_buffer_count
  }
  {
    r1:0 = C15:14                               // Control registers that keep processor cycle count (64-bits)
    memw(r5+#0) = r0                            // Save loop/function ID
  }
  {
    memw(r5+#8) = r1                            // Save upper 32 bits
    memw(r5+#4) = r0                            // Save lower 32 bits
  }
  .falign
.LBB0_3:                                        // Restore the registers from the stack
  {
    r1 = memw(r29+#0)                           // We will restore P3:0 but need an intermediate usual register (R1) that hasn't already been restored            
    r5:4 = memd(r29+#24)                        // Restore R5:4
  }
  {
    r3:2 = memd(r29+#16)                        // Restore R3:2
    p3:0 = r1                                   // Restore P3:0 (via R1, not yet restored)
  }
  {
    r1:0 = memd(r29+#8)                         // Restore R1:0
    dealloc_return                              // Deallocate the stack and return
  }
.Lfunc_end0:
  .size  lwp_handler, .Lfunc_end0-lwp_handler
