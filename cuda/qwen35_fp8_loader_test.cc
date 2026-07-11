// Host-only detection gate for Qwen3.5/3.6 block-FP8, ModelOpt, and compressed-tensors NVFP4.
#include "qwen35_fp8_loader.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

struct Entry { std::string name, dtype; std::vector<int64_t> shape; size_t bytes; };
static std::string build_st(const std::vector<Entry>& es) {
    std::string json="{"; size_t off=0; bool first=true; std::string data;
    for(const auto&e:es){
        if(!first) json+=",";
        first=false;
        json+="\""+e.name+"\":{\"dtype\":\""+e.dtype+"\",\"shape\":[";
        for(size_t i=0;i<e.shape.size();i++){ if(i)json+=","; json+=std::to_string(e.shape[i]); }
        json+="],\"data_offsets\":["+std::to_string(off)+","+std::to_string(off+e.bytes)+"]}";
        off+=e.bytes; data.append(e.bytes,'\0');
    }
    json+="}"; uint64_t n=json.size(); std::string out;
    out.append((const char*)&n,8); out+=json; out+=data; return out;
}
static void write_file(const std::string&p,const std::string&b){
    FILE*f=fopen(p.c_str(),"wb"); assert(f); assert(fwrite(b.data(),1,b.size(),f)==b.size()); fclose(f);
}
static qwen35fp8::Layout detect_case(const char *name,const char *quant_json,bool packed){
    std::string dir=std::string("/tmp/q35_loader_")+name;
    std::filesystem::remove_all(dir); std::filesystem::create_directories(dir);
    std::vector<Entry> es;
    for(int l=0;l<2;l++) es.push_back({"model.language_model.layers."+std::to_string(l)+
        ".input_layernorm.weight","BF16",{32},64});
    es.push_back({std::string("model.language_model.layers.0.mlp.experts.0.gate_proj.")+
                  (packed?"weight_packed":"weight"),packed?"U8":"F8_E4M3",{16,packed?16:32},512});
    write_file(dir+"/model.safetensors",build_st(es));
    write_file(dir+"/config.json",std::string("{\"model_type\":\"qwen3_5_moe\",")+quant_json+"}");
    st::Model m; std::string err; assert(m.open(dir.c_str(),err));
    qwen35fp8::Layout L;
    if(!qwen35fp8::detect(m,L,err)){ fprintf(stderr,"%s detect failed: %s\n",name,err.c_str()); abort(); }
    assert(L.n_layers==2); return L;
}
int main(){
    auto fp8=detect_case("fp8","\"quantization_config\":{\"quant_method\":\"fp8\"}",false);
    assert(!fp8.modelopt&&!fp8.compressed&&!fp8.mixed_nvfp4());
    auto mo=detect_case("modelopt","\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\"}",false);
    assert(mo.modelopt&&!mo.compressed&&mo.mixed_nvfp4());
    auto ct=detect_case("compressed","\"quantization_config\":{\"quant_method\":\"compressed-tensors\",\"format\":\"mixed-precision\",\"config_groups\":{\"g\":{\"format\":\"nvfp4-pack-quantized\"}}}",true);
    assert(!ct.modelopt&&ct.compressed&&ct.mixed_nvfp4());
    puts("qwen35 FP8/ModelOpt/compressed-tensors detect: OK");
    return 0;
}
