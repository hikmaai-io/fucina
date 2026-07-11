#include "model_plan.h"

#include <stdio.h>
#include <string>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static PlannedTensor tensor(const char *name, size_t bytes, size_t align) {
    PlannedTensor t;
    t.source.logical_name=name; t.source.source_name=std::string("model.")+name;
    t.source.dtype="BF16"; t.source.shape={4,8}; t.source.bytes=64;
    t.transform=TensorTransform::BF16_TO_F32; t.destination=WeightEncoding::F32;
    t.arena=AllocationClass::CORE_WEIGHTS; t.consumer="decode";
    t.bytes=bytes; t.alignment=align;
    return t;
}

int main() {
    std::string err;
    ModelPlan p;
    CHECK(p.add(tensor("a",33,32),err));
    CHECK(p.add(tensor("b",65,64),err));
    PlannedTensor alias=tensor("a_alias",1,1); alias.aliases=0; alias.bytes=0;
    CHECK(p.add(alias,err));
    CHECK(p.finalize(err));
    CHECK(p.tensors()[0].arena_offset==0);
    CHECK(p.tensors()[1].arena_offset==64);
    CHECK(p.tensors()[2].arena_offset==0);
    CHECK(p.bytes(AllocationClass::CORE_WEIGHTS)==129);
    const std::string once=p.json(), twice=p.json();
    CHECK(once==twice);
    CHECK(once.find("\"logical_name\":\"a_alias\"")!=std::string::npos);

    ModelPlan bad_shape; PlannedTensor bad=tensor("bad",4,4); bad.source.shape[0]=0;
    CHECK(!bad_shape.add(bad,err));
    ModelPlan bad_align; bad=tensor("bad_align",4,3);
    CHECK(!bad_align.add(bad,err));
    ModelPlan bad_alias; bad=tensor("bad_alias",4,4); bad.aliases=1;
    CHECK(bad_alias.add(bad,err));
    CHECK(!bad_alias.finalize(err));

    puts("model plan validation/alias/deterministic JSON: OK");
    return 0;
}
