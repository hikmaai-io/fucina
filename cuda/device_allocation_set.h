#ifndef FUCINA_DEVICE_ALLOCATION_SET_H
#define FUCINA_DEVICE_ALLOCATION_SET_H

#include <stddef.h>
#include <stdint.h>
#include <utility>
#include <vector>

struct DeviceAllocationOps {
    void *context = nullptr;
    int (*allocate)(void *context, void **ptr, size_t bytes) = nullptr;
    void (*release)(void *context, void *ptr) = nullptr;
    int (*upload)(void *context, void *dst, const void *src, size_t bytes) = nullptr;
};

struct DeviceAllocationRecord {
    void **slot = nullptr;
    void *pointer = nullptr;
    size_t bytes = 0;
    const char *label = nullptr;
};

class DeviceAllocationRegistry {
public:
    explicit DeviceAllocationRegistry(DeviceAllocationOps ops) : ops_(ops) {}
    DeviceAllocationRegistry(const DeviceAllocationRegistry &) = delete;
    DeviceAllocationRegistry &operator=(const DeviceAllocationRegistry &) = delete;
    ~DeviceAllocationRegistry() { reset(); }

    void reset() {
        for (size_t i = records_.size(); i > 0; --i) {
            DeviceAllocationRecord &record = records_[i - 1];
            if (record.pointer) ops_.release(ops_.context, record.pointer);
            if (record.slot && *record.slot == record.pointer) *record.slot = nullptr;
        }
        records_.clear(); total_bytes_ = 0;
    }

    size_t size() const { return records_.size(); }
    size_t bytes() const { return total_bytes_; }

private:
    friend class DeviceAllocationSet;
    void append(std::vector<DeviceAllocationRecord> &&records, size_t bytes) {
        records_.reserve(records_.size() + records.size());
        for (DeviceAllocationRecord &record : records) records_.push_back(record);
        total_bytes_ += bytes;
    }

    DeviceAllocationOps ops_;
    std::vector<DeviceAllocationRecord> records_;
    size_t total_bytes_ = 0;
};

class DeviceAllocationSet {
public:
    explicit DeviceAllocationSet(DeviceAllocationOps ops) : ops_(ops) {}
    DeviceAllocationSet(const DeviceAllocationSet &) = delete;
    DeviceAllocationSet &operator=(const DeviceAllocationSet &) = delete;
    ~DeviceAllocationSet() { rollback(); }

    bool allocate(void **slot, size_t bytes, const char *label) {
        if (committed_ || !slot || !ops_.allocate || !ops_.release || bytes == 0) return false;
        void *pointer = nullptr;
        if (ops_.allocate(ops_.context, &pointer, bytes) != 0 || !pointer) return false;
        *slot = pointer;
        records_.push_back({slot, pointer, bytes, label});
        total_bytes_ += bytes;
        return true;
    }

    bool adopt(void **slot, void *pointer, size_t bytes, const char *label) {
        if (committed_ || !slot || !pointer || bytes == 0 || *slot != pointer) return false;
        records_.push_back({slot, pointer, bytes, label});
        total_bytes_ += bytes;
        return true;
    }

    bool upload(void *destination, const void *source, size_t bytes) {
        if (committed_ || !ops_.upload || !destination || !source || bytes == 0) return false;
        bool owned = false;
        for (const DeviceAllocationRecord &record : records_) {
            const uintptr_t begin = (uintptr_t)record.pointer;
            const uintptr_t end = begin + record.bytes;
            const uintptr_t dst = (uintptr_t)destination;
            if (dst >= begin && dst <= end && bytes <= end - dst) { owned = true; break; }
        }
        return owned && ops_.upload(ops_.context, destination, source, bytes) == 0;
    }

    void rollback() {
        if (committed_) return;
        for (size_t i = records_.size(); i > 0; --i) {
            DeviceAllocationRecord &record = records_[i - 1];
            ops_.release(ops_.context, record.pointer);
            if (record.slot && *record.slot == record.pointer) *record.slot = nullptr;
        }
        records_.clear(); total_bytes_ = 0;
    }

    bool commit(DeviceAllocationRegistry &registry) {
        if (committed_) return false;
        registry.append(std::move(records_), total_bytes_);
        records_.clear(); total_bytes_ = 0; committed_ = true;
        return true;
    }

    size_t size() const { return records_.size(); }
    size_t bytes() const { return total_bytes_; }

private:
    DeviceAllocationOps ops_;
    std::vector<DeviceAllocationRecord> records_;
    size_t total_bytes_ = 0;
    bool committed_ = false;
};

#endif  // FUCINA_DEVICE_ALLOCATION_SET_H
