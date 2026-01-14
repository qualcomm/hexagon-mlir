//===- CroutonTypes.td ----------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_CROUTON_IR_CROUTON_TYPES_TD
#define HEXAGON_DIALECT_CROUTON_IR_CROUTON_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "hexagon/Dialect/Crouton/IR/CroutonDialect.td"

// Base class for crouton dialect types.
class Crouton_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Crouton_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = typeMnemonic;
}

def Crouton_ElemType : AnyTypeOf<[F16, I8], "crouton-element-type">;

// Define the enum StorageSpecifier kind attribute.
def Crouton_CroutonType : Crouton_Type<"Crouton", "crouton", [ShapedTypeInterface]> {
    let summary = "Crouton/crouton-table (`::mlir::crouton::CroutonType`)";

    let description = [{
        Crouton Type is a descriptor (much like MemRef) that contains table of
        pointers to 2KB memory chunk (crouton), along with meta-data on shape,
        element-type, and location.

        A crouton is often called crouton-table because each crouton type is not
        just a 2KB of buffer but a list of 2KB buffer pointers with shape meta-data
        on those pointers. It should be called crouton-vector because one could
        reference a particular crouton in a crouton-table by indexing into it. But
        we stick to crouton-table terminology nonetheless. The shape of the crouton
        type gives a relative (up-down, left-right) orientation to the 2KB memory
        chunks so that during computation IR should be able to query (indexing)
        spatial (view-like) locations of different croutons (e.g. crouton below a
        particular crouton).

        There is no layout information in a crouton as it could be re-assigned to
        different layouts over time.

        Arguments:

        shape: Ranked and statically shaped, it specifies the layout of crouton
               pointers e.g. 2x3 means 6 pointers that are to be interpreted as
               creating a 2x3 view (up-down, left, right), each pointer
               pointing to 2KB of memory.

        elementType: Specifies the element-type for the crouton.
                     Possible values: i8/f16

        vtcm: A crouton pointer may point to VTCM (attribute present) or DDR.
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType,
        "BoolAttr":$vtcm
    );

    let extraClassDeclaration = [{
        CroutonType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                              Type elementType) const {
            return CroutonType::get(shape.value_or(getShape()), elementType, getVtcm());
        }

        bool hasRank() const {
            return true;
        }

    }];

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "BoolAttr":$vtcm
        ), [{
            return $_get(elementType.getContext(), shape, elementType, vtcm);
        }]>,
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "Type":$elementType,
            "bool":$vtcm
        ), [{
            MLIRContext *ctx = elementType.getContext();
            BoolAttr vtcmAttr = BoolAttr::get(ctx, vtcm);
            return $_get(ctx, shape, elementType, vtcmAttr);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

#endif // HEXAGON_DIALECT_CROUTON_IR_CROUTON_TYPES_TD
