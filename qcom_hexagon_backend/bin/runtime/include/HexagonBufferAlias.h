#ifndef HEXAGONBUFFERALIAS_H_
#define HEXAGONBUFFERALIAS_H_

#include "HexagonBuffer.h"
#include <memory>
#include <vector>

struct Allocation;
namespace hexagon {

/// Represent an alias of an existing HexagonBuffer
///
/// Used when we create a crouton table alias of a flat
/// HexagonBuffer
class HexagonBufferAlias {
public:
  /// Allocate 1d (contiguous) memory within memory scopes.
  HexagonBufferAlias(HexagonBuffer &buffer, size_t nbytes);

  /// Destruction deallocates the underlying allocations.
  // ~HexagonBufferAlias();

  /// Prevent copy/move construction and assignment
  HexagonBufferAlias(const HexagonBufferAlias &) = delete;
  HexagonBufferAlias &operator=(const HexagonBufferAlias &) = delete;
  HexagonBufferAlias(HexagonBufferAlias &&) = delete;
  HexagonBufferAlias &operator=(HexagonBufferAlias &&) = delete;

  void *GetCroutonTableBase();
  std::vector<void *> GetCroutonTable();
  HexagonBuffer *origBuffer;

private:
  std::vector<void *> croutonTable;
};

} // namespace hexagon

#endif // HEXAGONBUFFERALIAS_H_
