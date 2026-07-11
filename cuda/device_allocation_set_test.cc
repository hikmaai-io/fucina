#include "device_allocation_set.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)

struct Fake {
    int calls=0, fail_at=0, uploads=0, upload_fail_at=0;
    std::vector<void*> live;
    std::vector<void*> released;
};
static int fake_alloc(void *ctx,void **out,size_t n){
    Fake *f=(Fake*)ctx; f->calls++;
    if(f->fail_at&&f->calls==f->fail_at) return 1;
    *out=malloc(n); if(!*out)return 1; f->live.push_back(*out); return 0;
}
static void fake_free(void *ctx,void *p){ Fake*f=(Fake*)ctx; f->released.push_back(p); free(p); }
static int fake_upload(void *ctx,void *dst,const void *src,size_t n){
    Fake*f=(Fake*)ctx; f->uploads++; if(f->upload_fail_at&&f->uploads==f->upload_fail_at)return 1;
    memcpy(dst,src,n); return 0;
}

int main(){
    Fake f; DeviceAllocationOps ops{&f,fake_alloc,fake_free,fake_upload};
    void *a=nullptr,*b=nullptr,*c=nullptr;
    f.fail_at=3;
    { DeviceAllocationSet tx(ops);
      CHECK(tx.allocate(&a,16,"a")); CHECK(tx.allocate(&b,32,"b"));
      int value=7; f.upload_fail_at=2;
      CHECK(tx.upload(a,&value,sizeof(value))); CHECK(!tx.upload(b,&value,sizeof(value)));
      CHECK(!tx.allocate(&c,64,"c")); CHECK(tx.size()==2); CHECK(tx.bytes()==48); }
    CHECK(!a&&!b&&!c); CHECK(f.released.size()==2);
    CHECK(f.released[0]==f.live[1]&&f.released[1]==f.live[0]);

    f=Fake{}; DeviceAllocationRegistry registry(ops);
    { DeviceAllocationSet tx(ops);
      CHECK(tx.allocate(&a,7,"a")); CHECK(tx.allocate(&b,9,"b"));
      c=malloc(5); CHECK(c); CHECK(tx.adopt(&c,c,5,"adopted"));
      CHECK(tx.commit(registry)); CHECK(!tx.commit(registry)); }
    CHECK(a&&b&&c&&registry.size()==3&&registry.bytes()==21&&f.released.empty());
    registry.reset(); CHECK(!a&&!b&&!c&&registry.size()==0&&registry.bytes()==0);
    CHECK(f.released.size()==3);
    puts("device allocation rollback/commit/registry teardown: OK");
    return 0;
}
