//===- Common.cpp - defines some common types and functions ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir {
namespace hexagon {

bool isMemorySpaceIntTypeOrDefault(MemRefType type, int &memSpaceID) {
  auto memSpace = type.getMemorySpace();
  if (memSpace) {
    if (!llvm::isa<IntegerAttr>(memSpace))
      return false;
    memSpaceID = type.getMemorySpaceAsInt();
  } else
    memSpaceID = DEFAULT_DDR_ADDRESS_SPACE; // Default memory-space
  return true;
}

bool isInVTCMAddressSpace(MemRefType type) {
  int addrSpace;
  if (isMemorySpaceIntTypeOrDefault(type, addrSpace)) {
    return addrSpace == VTCM_ADDRESS_SPACE;
  }
  return false;
}

bool isContiguousMemrefType(MemRefType type) {
  auto memrefType = dyn_cast<mlir::MemRefType>(type);

  return memrefType &&
         (memrefType.getLayout().isIdentity() ||
          (memrefType.getNumElements() > 0 &&
           memref::isStaticShapeAndContiguousRowMajor(memrefType)));
}

bool isStridedMultiDimMemrefType(mlir::MemRefType memRefType, int64_t &stride,
                                 int64_t &width) {
  // Extract the offset and strides from the type.
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(memRefType.getStridesAndOffset(strides, offset)))
    return false;

  auto isStaticStrideOrOffset = [](int64_t strideOrOffset) {
    return !ShapedType::isDynamic(strideOrOffset);
  };

  // TODO: support dynamic strides; offsets could be dynamic - we only require
  // static strides (stride[1] = 1, stride[0] > size[1])
  if (!llvm::all_of(strides,
                    isStaticStrideOrOffset)) // Each stride should be static
    return false;

  int64_t rank = memRefType.getRank();
  if (rank <= 1 || strides[rank - 1] != 1) {
    return false;
  }

  // if stride of second inner dimension is greater than size of inner
  // dimension, then we have non-contiguous 2D access
  if (strides[rank - 2] > memRefType.getShape()[rank - 1]) {
    stride = strides[rank - 2];
    width = memRefType.getShape()[rank - 1];
    return true;
  }

  return false;
}

void addTypeConversions(MLIRContext *context,
                        LLVMTypeConverter &typeConverter) {
  // The hexagon llvm backend does not define a separate address
  // address space for VTCM and DDR (conceptually ptr<0>, ptr<1>).
  // In our mlir flow we allocate on VTCM using runtime hexagonmem
  // calls which return memref<__, 1> that lower to ptr<1>. To be
  // compatible with llvm hexagon backend, at the last stage of
  // lowering we collapse DDR_ADDRESS_SPACE.
  typeConverter.addTypeAttributeConversion(
      [context](BaseMemRefType type, IntegerAttr vtcmAttr) {
        // Only value 1 is accepted as VTCM memory space
        if (vtcmAttr.getInt() == hexagon::VTCM_ADDRESS_SPACE) {
          return IntegerAttr::get(IntegerType::get(context, 64),
                                  hexagon::DEFAULT_DDR_ADDRESS_SPACE);
        }
        return vtcmAttr;
      });
}

} // namespace hexagon
} // namespace mlir
