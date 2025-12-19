## Hexagon API runtime memory modules: Class Diagram

The diagram represents runtime memory modules for a Hexagon API, which includes several classes:

1. **HexagonAPI**:
   - Manages global resources and provides access to the `VtcmPool` and `BufferManager`.
   - Contains methods to acquire and release runtime resources.
   > TODO: HexagonAPI can have ThreadManager, UserDMA and PowerManager as well

2. **VtcmPool**:
   - Manages VTCM (Very Tightly Coupled Memory) allocations.
   - Provides methods to allocate and free VTCM memory, and track the total and allocated bytes.

3. **BufferManager**:
   - Manages Hexagon buffers, including allocation and deallocation.
   - Keeps a map of allocated buffers and provides methods to copy data between buffers.

4. **Allocation**:
   - Represents a generic memory allocation with a specified size and alignment.

5. **DDRAllocation** and **VTCMAllocation**:
   - Specialized types of `Allocation` for DDR (Double Data Rate) and VTCM memory, respectively.

6. **HexagonBuffer**:
   - Represents a buffer allocated within a specific memory scope.
   - Can handle both flat and crouton memory allocations.
   - Provides methods to get the data pointer, address space, and copy data from another buffer.

The relationships between these classes show how `HexagonAPI` interacts with `BufferManager` and `VtcmPool`, and how `BufferManager` manages `HexagonBuffer` instances. The `HexagonBuffer` class is linked to `Allocation`, which is further specialized into `DDRAllocation` and `VTCMAllocation`.

```mermaid
classDiagram
    class HexagonAPI {
        +static HexagonAPI* Global() // Retrieve the global singleton instance of the HexagonAPI.
        +HexagonAPI()
        +~HexagonAPI()
        +void AcquireResources() // Ensures all runtime resources are acquired.
        +void ReleaseResources() // Ensures all runtime resources are freed.
        +void *Alloc(size_t nbytes, bool isVtcm); // Allocate a single, contiguous memory region.
        +void *Alloc(size_t nallocs, size_t nbytes, bool isVtcm); // Allocate the region(s) needed for Hexagon's indirect-tensor format.
        +void Free(void *ptr); // Frees the allocated memory region.
        +void Copy(void *dst, void *src, size_t nbytes); // Copies the data from source src into destination dst.
        +VtcmPool* getVtcmPool()
        +BufferManager* getBufferManager()
        TODO: +ThreadManager* getThreadManager()
        TODO: +UserDMA* getUserDMA()
        TODO: +PowerManager* getPowerManager()
        -std::unique_ptr&lt;BufferManager> bufferManager
        -std::unique_ptr&lt;VtcmPool> runtimeVtcm
        TODO: -std::unique_ptr&lt;ThreadManager> threadManager
        TODO: -std::unique_ptr&lt;UserDMA> userDma
        TODO: -std::unique_ptr&lt;PowerManager> powerManager
    }
    class VtcmPool {
        +VtcmPool() // Allocates VTCM memory, and manages runtime allocations.
        +~VtcmPool() // Destruction deallocates the underlying VTCM allocation.
        +void* Allocate(size_t nbytes)
        +void Free(void* ptr, size_t nbytes)
        +size_t VtcmDeviceBytes() // Returns the total number of bytes in this pool.
        +size_t VtcmAllocatedBytes() // Returns the total allocated bytes in this pool.
        +bool IsVtcm(void* ptr, unsigned size)
        -unsigned int vtcmDeviceSize_
        -unsigned int vtcmAllocatedSize_
        -void* vtcmData_
        -std::vector<std::pair<char*, size_t>> allocations_
        -std::vector<std::pair<char*, size_t>> free_
    }
    class BufferManager {
        +~BufferManager()
        +void FreeHexagonBuffer(void* ptr)
        void* AllocateHexagonBuffer(Args&&... args)
        +HexagonBuffer* FindHexagonBuffer(void* ptr)
        -std::unordered_map&lt;void*, std::unique_ptr&lt;HexagonBuffer>> bufferMap_
        +void Copy(void* src, void* dst, size_t nbytes)
    }
    class Allocation {
        +Allocation(size_t allocation_nbytes, size_t alignment)
        +~Allocation()
        +void* data_
        +size_t allocation_nbytes_
        +size_t alignment_
    }

    class DDRAllocation {
        +DDRAllocation(size_t nbytes, size_t alignment)
        +~DDRAllocation()
    }

    class VTCMAllocation {
        +VTCMAllocation(size_t nbytes, size_t alignment)
        +~VTCMAllocation()
    }

    class MemoryCopy {
        +MemoryCopy(void *dest, void *src, size_t numBytes)
        +bool IsDirectlyBefore(const MemoryCopy &other)
        +static std::vector<MemoryCopy> MergeAdjacent(std::vector<MemoryCopy> microCopies)
        +void *dest;
        +void *src;
        +size_t numBytes;
    }

    class BufferSet {
        +static std::vector<MemoryCopy> MemoryCopies(const BufferSet &dest, const BufferSet &src, size_t bytesToCopy);
        +BufferSet(void *const *buffers, size_t numRegions, size_t regionSizeBytes)
        +size_t TotalBytes()
        +void *const *buffers;
        +size_t numRegions;
        +size_t regionSizeBytes;
    }

    class HexagonBuffer {
        // Allocate flat memory within the memory scope
        +HexagonBuffer(size_t nbytes, size_t alignment, Optional&lt;String> scope)

        // Allocate crouton memory within the memory scope
        +HexagonBuffer(size_t nallocs, size_t nbytes, size_t alignment, Optional&lt;String> scope)
        +~HexagonBuffer()
        +void* GetPointer() // Return data pointer into the buffer
        +AddressSpace GetAddressSpace() const // Return address space of the allocation.
        +void CopyTo(void *data, size_t nbytes) const; // Copy data from a Hexagon Buffer an external buffer.
        +void CopyFrom(void *data, size_t nbytes); // Copy data from an external buffer to a Hexagon Buffer.
        +void CopyFrom(const HexagonBuffer &other, size_t nbytes); // Copy data from one Hexagon Buffer to another.
        +void hexagonBufferCopyAcrossRegions(const BufferSet &dest, const BufferSet &src, size_t bytesToCopy, bool srcIsHexbuff, bool destIsHexbuff);
        -size_t TotalBytes() const
        -void SetAddressSpace(Optional&lt;String> scope)
        -std::vector&lt;void*> allocations_
        -std::vector&lt;std::unique_ptr&lt;Allocation>> managed_allocations_
        -size_t ndim_
        -size_t nbytes_per_allocation_
        -AddressSpace address_space_
    }

    HexagonBuffer --> Allocation
    HexagonBuffer --> BufferSet
    BufferSet --> MemoryCopy
    Allocation <|-- DDRAllocation
    Allocation <|-- VTCMAllocation
    BufferManager --> HexagonBuffer
    HexagonAPI --> BufferManager
    HexagonAPI --> VtcmPool
```

## Hexagon API runtime memory modules: Sequence Diagram

The sequence diagram illustrates the interactions between various components in the Hexagon API runtime modules during allocation, copying, and deallocation of memory buffers.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant C_API
    participant HexagonAPI
    participant BufferManager
    participant HexagonBuffer    

    User ->> C_API : alloc()
    C_API ->> HexagonAPI: Alloc()    
    HexagonAPI ->> BufferManager: AllocateHexagonBuffer(..)
    BufferManager->>HexagonBuffer: HexagonBuffer(..)
    HexagonBuffer-->>BufferManager: HexagonBuffer instance
    BufferManager -->> HexagonAPI: void* ptr
    HexagonAPI -->> User: void* ptr
    User ->> C_API : copy(src, dst)
    C_API ->> HexagonAPI: Copy(src, dst)
    HexagonAPI ->> BufferManager: Copy(src, dst)
    BufferManager-->>BufferManager: FindHexagonBuffer(..)
    BufferManager->>HexagonBuffer: CopyFrom(src, nbytes)
    HexagonBuffer-->>BufferManager: Data copied
    User ->> C_API : dealloc(ptr)
    C_API ->> HexagonAPI: Free(ptr)
    HexagonAPI ->> BufferManager: FreeHexagonBuffer(ptr)
    BufferManager-->>BufferManager: FindHexagonBuffer(..)
    BufferManager->>HexagonBuffer: ~HexagonBuffer()
    HexagonBuffer-->>BufferManager: Deallocate resources
```

## Hexagon Buffer Allocation: Sequence Diagram

The sequence diagram outlines the process of allocating memory for a HexagonBuffer based on its address space, either DDR or VTCM.

```mermaid
sequenceDiagram
    participant BufferManager
    participant HexagonBuffer
    participant DDRAllocation
    participant VTCMAllocation
    participant VTCMPool

    BufferManager->>HexagonBuffer: new HexagonBuffer(nbytes, alignment, scope)
    HexagonBuffer->>HexagonBuffer: SetAddressSpace(scope)
    alt scope is DDR
        HexagonBuffer->>DDRAllocation: Allocator<AddressSpace::kDDR>(nbytes, alignment)
        DDRAllocation->>HexagonBuffer: Return allocation
    else scope is VTCM
        HexagonBuffer->>VTCMAllocation: Allocator<AddressSpace::kVTCM>(nbytes, alignment)
        VTCMAllocation->>VTCMPool: Allocate from VTCM pool
        VTCMPool->>VTCMAllocation: Return VTCM allocation
        VTCMAllocation->>HexagonBuffer: Return allocation
    end
    HexagonBuffer->>BufferManager: Return HexagonBuffer instance

```
